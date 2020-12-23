from Omniglot import Omniglot
import yaml
import argparse
import numpy as np

from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms.functional import *
import fasttext
import pickle
import random

import argparse

# python train_mapper_and_discriminator.py --embd_random_samples ./randomSamples_fromEmbeddings/sample_fasttext_embs_en_128.pickle --vae_ckp_path ./model_checkpoints/final_model_checkpoint.ckpt --config_path ./configs/bbvae_CompleteRun.yaml

#python logogram_language_generator.py --embd_random_samples ./randomSamples_fromEmbeddings/sample_fasttext_embs_en_128.pickle 
#--vae_ckp_path ./model_checkpoints/final_model_checkpoint.ckpt  --config_path ./configs/bbvae_CompleteRun.yaml 
# --export_path ./outputs/ --words_path words.txt --norm_option standard 
# --emb_bins_path ./model_checkpoints/cc_bins_128/

parser = argparse.ArgumentParser(description='Mapper training for feeding embeddings to VAE')
parser.add_argument('--embd_random_samples', help =  'path to the embedding samples', default='/content/drive/My Drive/vae/logogram-language-generator-master/sample_fasttext_embs.pickle')
parser.add_argument("--vae_ckp_path", type=str, default="logs/BetaVAE_B_setup2_run2/final_model_checkpoint.ckpt", help="checkpoint path of vae")
parser.add_argument("--config_path", type=str, default="./configs/bbvae_setup2.yaml", help="config")
parser.add_argument("--export_path", type=str, default="./vae_with_disc/", help="export")
parser.add_argument("--umwe_mappers_path", type=str, default="./model_checkpoints/", help="")
parser.add_argument("--test_onHelloWorld", type=bool, default=False, help="")
parser.add_argument("--emb_vector_dim", type=int, default=300, help="")
parser.add_argument("--vae_latent_dim", type=int, default=128, help="")
parser.add_argument("--mapper_numlayer", type=int, default=3, help="")

parser.add_argument("--words_path", type=str, default="./words/", help="")
parser.add_argument("--emb_bins_path", type=str, default="./model_checkpoints/cc_bins_128/", help="")
parser.add_argument("--siamese_path", type=str, default="./model_checkpoints/siamese.ckpt", help="")


args = parser.parse_args()

args.langs = ["en", "fr", "es", "it", "tr"]
# args.langs = ["en", "fr"]
args.emb_bins = {}
for lang in args.langs:
    ft = fasttext.load_model(args.emb_bins_path+"cc."+lang+".128.bin")
    ft.get_dimension()
    args.emb_bins[lang] = ft
    print(lang, " bins loaded")


device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

args.emb_mappers = {}
for lang in args.langs:
    if lang != "en":
        args.emb_mappers[lang] = torch.from_numpy(torch.load(args.umwe_mappers_path+"best_mapping_"+lang+"2en.t7")).to(device)


class MultilingualMapper(nn.Module):
    def __init__(self, device, embedding_vector_dim = 300, decoder_input_dim=128):
        super(MultilingualMapper, self).__init__()
        self.device = device
        self.embedding_vector_dim = embedding_vector_dim
        self.decoder_input_dim = decoder_input_dim
        self.mapper_numlayer = args.mapper_numlayer

        self.linear_layers = []
        self.batch_norms = []
        for layer in range(0, self.mapper_numlayer-1):
            self.linear_layers.append(nn.Linear(embedding_vector_dim, embedding_vector_dim))
            self.batch_norms.append(nn.BatchNorm1d(embedding_vector_dim))

        # final layer
        self.linear_layers.append(nn.Linear(embedding_vector_dim, decoder_input_dim))
        self.batch_norms.append(nn.BatchNorm1d(decoder_input_dim))


        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.batch_norms = nn.ModuleList(self.batch_norms)
        self.relu = nn.ReLU()

        self.bce = nn.BCEWithLogitsLoss()
      
    def forward(self, embedding_vector):
        inp = embedding_vector
        for layer in range(self.mapper_numlayer):
            out = self.linear_layers[layer](inp)
            out = self.batch_norms[layer](out)
            out = self.relu(out)
            inp = out        
        return out

    def triplet_loss(self, sameWords_diffLangs, diffWords_sameLangs):
        return self.bce(sameWords_diffLangs, torch.ones(sameWords_diffLangs.shape).to(self.device)) + self.bce(diffWords_sameLangs, torch.zeros(diffWords_sameLangs.shape).to(self.device))



class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 10),  # 64@96*96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64@48*48
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),    # 128@42*42
            nn.MaxPool2d(2),   # 128@21*21
            nn.Conv2d(128, 128, 4),
            nn.ReLU(), # 128@18*18
            nn.MaxPool2d(2), # 128@9*9
            nn.Conv2d(128, 256, 4),
            nn.ReLU(),   # 256@6*6
            nn.Flatten(),
            nn.Linear(9216, 4096),
            nn.Sigmoid()
        )

        self.classifier = nn.Linear(4096, 1)
        self.bce_fn = nn.BCEWithLogitsLoss()

    def forward(self, x1, x2):
        z1 = self.encoder.forward(x1)
        z2 = self.encoder.forward(x2)

        z = torch.abs(z1 - z2)
        y = self.classifier.forward(z)
        return y

    def triplet_loss(self, yp, yn):
        return self.bce_fn(yp, torch.ones(yp.shape).cuda()) + self.bce_fn(yn, torch.zeros(yn.shape).cuda())

    def train_triplet(self, loader, epochs=35):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=0.00001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

        for epoch in range(epochs):
            total_loss = 0
            for i,(xa,xp,xn) in enumerate(loader):
                yp = self.forward(xa.cuda(), xp.cuda())
                yn = self.forward(xa.cuda(), xn.cuda())

                loss = self.triplet_loss(yp, yn)

                loss.backward()
                optimizer.step()
                
                total_loss += loss

            print("Epoch: %d Loss: %f" % (epoch+1, total_loss.item()/len(loader)))
            scheduler.step()

        torch.save({'model_state_dict': self.state_dict()}, "siamese.ckpt")

    def eval_triplet(self, loader):
        self.eval()
        correct,total = 0,0

        for i,(xa,xp,xn) in enumerate(loader):
            yp = self.forward(xa.cuda(), xp.cuda())
            yn = self.forward(xa.cuda(), xn.cuda())

            correct += torch.sum(yp>0) + torch.sum(yn<0)
            total += yp.shape[0] + yn.shape[0]

            if i==100:
                break

        print("Accuracy: %f" % (correct/total).item())






def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_dataLoader(batch_size):
    transform = data_transforms()
    dataset = Omniglot(split="train", transform=transform)
    num_train_imgs = len(dataset)
    return DataLoader(dataset,
                      batch_size= batch_size,
                      shuffle = True,
                      drop_last=True)


def data_transforms():
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X))
    transform = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])
    return transform

def post_proccess(imgs):
    imgs = list(map(lambda x:resize(to_pil_image(x),105), imgs))
    imgs = list(map(lambda x:adjust_saturation(x,1.5), imgs))
    imgs = list(map(lambda x:adjust_gamma(x,1.5), imgs))
    imgs = list(map(lambda x:adjust_contrast(x,2), imgs))
    imgs = list(map(lambda x:to_tensor(x), imgs))
    for x in imgs:
        x[x>0.5]=1
    for x in imgs:
        x[x<0.4]=0

    #save_image(imgs, 'img3.png')
    return imgs[0].to(device)

def randomSample_from_embeddings(batch_size):
    with open(args.embd_random_samples+"sample_fasttext_embs_"+"en"+"_128.pickle", "rb") as f:
        random_sample = pickle.load(f)
    if batch_size == -1:
        batch_size = len(list(random_sample.values()))
    fake_batch = random.sample(list(random_sample.values()), batch_size)
    fake_batch = torch.stack(fake_batch, dim=0)
    return fake_batch
  
def normalize_embeddings(embeddings):
    normalized_embs = []

    for word in range(embeddings.shape[0]):
        sample_embeds = randomSample_from_embeddings(-1).to(device)
        std, mean = torch.std_mean(input=sample_embeds, dim=0, unbiased=True)    
        normalized_embs.append((embeddings[word,:] - mean) / std)
    normalized_embs = torch.stack(normalized_embs, dim=0)

    return normalized_embs.to(device)
  
def sameWords_from_diffLangs(batch_size):
    two_langs = random.sample(args.langs, 2)
    print(two_langs)
    with open(args.words_path+two_langs[0]+".txt") as f:
        words_1 = f.readlines()
    with open(args.words_path+two_langs[1]+".txt") as f:
        words_2 = f.readlines()
    words_1, words_2 = zip(*random.sample(list(zip(words_1, words_2)), batch_size))
    words_1 = list(words_1)
    words_2 = list(words_2)
    
    for w in range(len(words_1)):
        words_1[w] = words_1[w].split("\n")[0]
        words_2[w] = words_2[w].split("\n")[0]

    embs1 = []
    embs2 = []
    for word in range(int(batch_size/2)):
        emb1 = torch.from_numpy(args.emb_bins[two_langs[0]].get_word_vector(words_1[word].split("\n")[0])).to(device)
        emb2 = torch.from_numpy(args.emb_bins[two_langs[1]].get_word_vector(words_2[word].split("\n")[0])).to(device)
        
        if two_langs[0] != "en":
            emb1 = torch.matmul(emb1, args.emb_mappers[two_langs[0]])
        if two_langs[1] != "en":
            emb2 = torch.matmul(emb2, args.emb_mappers[two_langs[1]])
        embs1.append(emb1)
        embs2.append(emb2)
    
    normalized_embs1 = normalize_embeddings(torch.stack(embs1, dim=0))
    normalized_embs2 = normalize_embeddings(torch.stack(embs2, dim=0))

    return normalized_embs1, normalized_embs2

def diffWords_sameLangs(batch_size):
    one_lang = random.sample(args.langs, 1)[0]
    print(one_lang)
    with open(args.words_path+one_lang+".txt") as f:
        words_1 = random.sample(f.readlines(), batch_size)
    embs1 = []
    for word in range(batch_size):
        wrd = words_1[word].split("\n")[0]
        emb1 = torch.from_numpy(args.emb_bins[one_lang].get_word_vector(wrd)).to(device)
        if one_lang != "en":
            emb1 = torch.matmul(emb1, args.emb_mappers[one_lang])
        embs1.append(emb1)

    normalized_embs = normalize_embeddings(torch.stack(embs1, dim=0))

    return torch.chunk(normalized_embs, 2, dim=0)


def trainer(vae_model, mapper, siamese, batch_size, device):
    mapper_losses = []
    iters = 0
    num_epochs = 10
    epoch_size = 10

    # Initialize BCELoss function
    lr = 0.00001
    beta1 = 0.5
    # Setup Adam optimizers for both G and D
    optimizerM = optim.Adam(mapper.parameters(), lr=lr, betas=(beta1, 0.999))


    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        for i in range(epoch_size):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            mapper.zero_grad()

            # Format batch
            # positives
            x_anchor, x_positive = sameWords_from_diffLangs(batch_size)
            x_anchor = mapper(x_anchor).to(device)            
            x_positive = mapper(x_positive).to(device)
            x_anchor = normalize_embeddings(x_anchor)
            x_positive = normalize_embeddings(x_positive)
            x_anchor = vae_model.decode(x_anchor)
            x_positive = vae_model.decode(x_positive)

            # negatives
            x_anchor_2, x_negative = diffWords_sameLangs(batch_size)
            x_anchor_2 = mapper(x_anchor_2).to(device)
            x_negative = mapper(x_negative).to(device)
            x_anchor_2 = normalize_embeddings(x_anchor_2)
            x_negative = normalize_embeddings(x_negative)
            x_anchor_2 = vae_model.decode(x_anchor_2)
            x_negative = vae_model.decode(x_negative)

            processed_x_anchor = []
            processed_x_positive = []
            processed_x_anchor_2 = []
            processed_x_negative = []
            for i in range(int(batch_size/2)):
                processed_x_anchor.append  (post_proccess(torch.reshape(  x_anchor.data[i,:,:], [1,64,64]).cpu()))
                processed_x_positive.append(post_proccess(torch.reshape(x_positive.data[i,:,:], [1,64,64]).cpu()))
                processed_x_anchor_2.append(post_proccess(torch.reshape(x_anchor_2.data[i,:,:], [1,64,64]).cpu()))
                processed_x_negative.append(post_proccess(torch.reshape(x_negative.data[i,:,:], [1,64,64]).cpu()))
                
                
            processed_x_anchor = torch.stack(processed_x_anchor, dim=0)
            processed_x_positive = torch.stack(processed_x_positive, dim=0)
            processed_x_anchor_2 = torch.stack(processed_x_anchor_2, dim=0)
            processed_x_negative = torch.stack(processed_x_negative, dim=0)                


            must_be_similar = siamese(processed_x_anchor, processed_x_positive)
            must_be_different = siamese(processed_x_anchor_2, processed_x_negative)

            # Calculate loss on negative and positive examples
            loss = mapper.triplet_loss(must_be_similar, must_be_different)
            # Calculate gradients for mapper in backward pass
            loss.backward()
            optimizerM.step()



            # Output training stats
            print('[%d/%d][%d/%d]\tMapper_Loss: %.4f'%(epoch, num_epochs, i, epoch_size, loss.item()))

            # Save Losses for plotting later
            mapper_losses.append(loss.item())
            
            # Check how the generator is doing by saving G's output on fixed_noise
            # if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            #     randomsamples = []                
            #     with torch.no_grad():
            #       mapped = mapper(torch.randn(batch_size,args.emb_vector_dim).to(device))
            #     randomsamples.append(mapped.to(device))

            #     embed_samples = []
            #     one_embed = randomSample_from_embeddings(batch_size).to(device)
            #     with torch.no_grad():
            #       mapped_embed = mapper(one_embed).to(device)
            #     embed_samples.append(mapped_embed)

            #     randomsamples = torch.stack(randomsamples, dim=0)
            #     recons_randoms = vae_model.decode(randomsamples)

            #     embedsamples = torch.stack(embed_samples, dim=0)
            #     recons_embeds = vae_model.decode(embedsamples)

            #     vutils.save_image(recons_randoms.data,
            #                   f"{args.export_path}test_random{i}.png",
            #                   normalize=True,
            #                   nrow=12)
            #     vutils.save_image(recons_embeds.data,
            #                   f"{args.export_path}test_embed{i}.png",
            #                   normalize=True,
            #                   nrow=12)
            # iters += 1
    return mapper

def load_vae_model(vae_checkpointPath, config):
    model = vae_models[config['model_params']['name']](**config['model_params'])
    experiment = VAEXperiment(model,
                          config['exp_params'])


    checkpoint = torch.load(vae_checkpointPath, map_location=lambda storage, loc: storage)
    new_ckpoint = {}
    for k in checkpoint["state_dict"].keys():
      newKey = k.split("model.")[1]
      new_ckpoint[newKey] = checkpoint["state_dict"][k]

    model.load_state_dict(new_ckpoint)
    model.eval()
    return model

def load_siamese():

    model = Siamese().cuda()
    model.load_state_dict(torch.load(args.siamese_path))
    model.eval()
    return model

def main():
    print("on main")
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    batch_size = 64
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(device)
    
    vae_model = load_vae_model(args.vae_ckp_path, config).to(device)

    mapper = MultilingualMapper(device, args.emb_vector_dim, args.vae_latent_dim).to(device)
    mapper.apply(weights_init)

    siamese = load_siamese()
    mapper = trainer(vae_model=vae_model, mapper=mapper, siamese=siamese, batch_size=batch_size, device=device)

    torch.save(mapper.state_dict(), args.export_path+"multilingualMapper_checkpoint_1layered.pt")
    


    if args.test_onHelloWorld:
        with open("/content/drive/My Drive/vae/logogram-language-generator-master/fasttext_hello_world.pickle", "rb") as f:
          helloworld = pickle.load(f)

        hello = torch.from_numpy(helloworld["hello"]).to(device)
        world = torch.from_numpy(helloworld["world"]).to(device)


        helloworld = torch.stack([hello, world], dim=0).to(device)

        embed_samples = []
        with torch.no_grad():
          mapped_embed = mapper(helloworld).to(device)
        embed_samples.append(mapped_embed)
        embedsamples = torch.stack(embed_samples, dim=0)
        recons_embeds = vae_model.decode(embedsamples)
        vutils.save_image(recons_embeds.data,
                      args.export_path+"helloWorld.png",
                      normalize=True,
                      nrow=12)




    print("ALL DONE!")





if __name__ == "__main__":
    main()

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
parser.add_argument("--test_onHelloWorld", type=bool, default=False, help="")
parser.add_argument("--emb_vector_dim", type=int, default=300, help="")
parser.add_argument("--vae_latent_dim", type=int, default=128, help="")
parser.add_argument("--mapper_numlayer", type=int, default=3, help="")


args = parser.parse_args()


class EmbeddingMapping(nn.Module):
    def __init__(self, device, embedding_vector_dim = 300, decoder_input_dim=128):
        super(EmbeddingMapping, self).__init__()
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
      
    def forward(self, embedding_vector):
        inp = embedding_vector
        for layer in range(self.mapper_numlayer):
            out = self.linear_layers[layer](inp)
            out = self.batch_norms[layer](out)
            out = self.relu(out)
            inp = out        
        return out

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.ndf = 64
        self.nc = 1
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


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


def randomSample_from_embeddings(batch_size):
    with open(args.embd_random_samples, "rb") as f:
        random_sample = pickle.load(f)

    fake_batch = random.sample(list(random_sample.values()), batch_size)
    fake_batch = torch.stack(fake_batch, dim=0)
    return fake_batch
    




def trainer(vae_model, mapper, netD, batch_size, device):
    # Training Loop
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    num_epochs = 10
    dataloader = get_dataLoader(batch_size=batch_size)

    # Initialize BCELoss function
    criterion = nn.BCELoss()
    lr = 0.00001
    beta1 = 0.5
    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(mapper.parameters(), lr=lr, betas=(beta1, 0.999))



    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = randomSample_from_embeddings(batch_size).to(device)

    criterion = nn.BCELoss()

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.


    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            embeddings = randomSample_from_embeddings(batch_size).to(device)
            # Generate fake image batch with G
            fake = mapper(embeddings).to(device)
            fake = vae_model.decode(fake)

            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            mapper.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                randomsamples = []                
                with torch.no_grad():
                  mapped = mapper(torch.randn(batch_size,args.emb_vector_dim).to(device))
                randomsamples.append(mapped.to(device))

                embed_samples = []
                one_embed = randomSample_from_embeddings(batch_size).to(device)
                with torch.no_grad():
                  mapped_embed = mapper(one_embed).to(device)
                embed_samples.append(mapped_embed)

                randomsamples = torch.stack(randomsamples, dim=0)
                recons_randoms = vae_model.decode(randomsamples)

                embedsamples = torch.stack(embed_samples, dim=0)
                recons_embeds = vae_model.decode(embedsamples)

                vutils.save_image(recons_randoms.data,
                              f"{args.export_path}test_random{i}.png",
                              normalize=True,
                              nrow=12)
                vutils.save_image(recons_embeds.data,
                              f"{args.export_path}test_embed{i}.png",
                              normalize=True,
                              nrow=12)

            iters += 1
    return vae_model, mapper, netD

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

    mapper = EmbeddingMapping(device, args.emb_vector_dim, args.vae_latent_dim).to(device)
    mapper.apply(weights_init)

    netD = Discriminator(device).to(device)
    netD.apply(weights_init)
    vae_model, mapper, netD = trainer(vae_model=vae_model, mapper=mapper, netD=netD, batch_size=batch_size, device=device)

    torch.save(mapper.state_dict(), args.export_path+"mapper_checkpoint.pt")
    torch.save(netD.state_dict(), args.export_path+"discriminator_checkpoint.pt")
    


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

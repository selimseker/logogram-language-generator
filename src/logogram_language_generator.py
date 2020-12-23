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
import imageio
from scipy.interpolate import interp1d
import pickle
import random
import fasttext
from torchvision.transforms.functional import *
from PIL import Image, ImageFont, ImageDraw
from MapperLayer import umwe2vae

from MapperLayer import EmbeddingMapping
from MapperLayer import MultilingualMapper
import matplotlib.pyplot as plt
import math

parser = argparse.ArgumentParser(description='Logogram Language Generator Main script')
parser.add_argument('--embd_random_samples', help =  'path to the embedding samples', default='./randomSamples_fromEmbeddings/')
parser.add_argument("--vae_ckp_path", type=str, default="./model_checkpoints/final_model_checkpoint.ckpt", help="checkpoint path of vae")
parser.add_argument("--config_path", type=str, default="./configs/bbvae_CompleteRun.yaml", help="config")
parser.add_argument("--export_path", type=str, default="./outputs/", help="export")

# parser.add_argument("--test_onHelloWorld", type=bool, default=True, help="")
# parser.add_argument("--emb_vector_dim", type=int, default=300, help="")
# parser.add_argument("--vae_latent_dim", type=int, default=128, help="")
# parser.add_argument("--mapper_numlayer", type=int, default=3, help="")

parser.add_argument("--umwe_mappers_path", type=str, default="./model_checkpoints/", help="config")
parser.add_argument("--words_path", type=str, default="./words.txt", help="config")

 # "standard" for statistical normalization "mapper" for linear layered mapper trained with discriminator 
parser.add_argument("--norm_option", type=str, default="standard", help="config")
parser.add_argument("--mapper_layer_ckp", type=str, default="./lossy_mapper/umwe2vae.ckpt", help="config")


parser.add_argument("--emb_bins_path", type=str, default="./model_checkpoints/cc_bins_128/", help="config")

parser.add_argument("--gif", type=str, default="True", help="config")

parser.add_argument("--siamese_mapper", type=str, default="False", help="config")


args = parser.parse_args()
with open(args.words_path, "r") as f:
    args.words = f.readlines()
for word in range(len(args.words)):
    args.words[word] = args.words[word].split("\n")[0]

args.latent_dim = 128
args.emb_vector_dim = 128

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


def randomSample_from_embeddings(batch_size, lang):
    with open(args.embd_random_samples+"sample_fasttext_embs_"+lang+"_128.pickle", "rb") as f:
        random_sample = pickle.load(f)
    if batch_size == -1:
        batch_size = len(list(random_sample.values()))
    fake_batch = random.sample(list(random_sample.values()), batch_size)
    fake_batch = torch.stack(fake_batch, dim=0)
    return fake_batch
    
def load_vae_model(vae_checkpointPath):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
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

def load_mapper(vae_model, mapper_type="mapper-disc"):
    if mapper_type == "mapper-disc":
        model = EmbeddingMapping(device, args.emb_vector_dim, args.latent_dim)
        model.load_state_dict(torch.load(args.mapper_layer_ckp))
        model.eval()
    elif mapper_type == "mapper-lossy":
        model = umwe2vae(vae_model, in_dim=128, out_dim=128)
        model.load_state_dict(torch.load(args.mapper_layer_ckp))   
        model.eval() 
    elif mapper_type == "siamese_mapper":
        model = MultilingualMapper(device, args.emb_vector_dim, args.latent_dim)
        model.load_state_dict(torch.load("./layerWithSiamese/multilingualMapper_checkpoint.pt"))
        model.eval()
    return model.to(device)

def embed_words():
    embs = []
    lang = ""
    for word in args.words:
        if word.split(" - ")[1] != lang:
            lang = word.split(" - ")[1]
            ft = fasttext.load_model(args.emb_bins_path+"cc."+lang+".128.bin")
            ft.get_dimension()
        emb = torch.from_numpy(ft.get_word_vector(word.split(" - ")[0])).to(device)
        if lang != "en":
            multilingual_mapper = torch.from_numpy(torch.load(args.umwe_mappers_path+"best_mapping_"+lang+"2en.t7")).to(device)
            emb = torch.matmul(emb, multilingual_mapper)
        embs.append(emb)
    embs = torch.stack(embs, dim=0)
    return embs
    # numpy to tensor word embs
    # torch.from_numpy(helloworld["hello"]).to(device)

    # best_mapping = "./best_mapping_fr2en.t7"
    # Readed_t7 = torch.from_numpy(torch.load(best_mapping)).to(device)
    # mapped_hello = torch.matmul(hello, Readed_t7)
    # mapped_world = torch.matmul(world, Readed_t7)


def normalize_embeddings(embeddings, vae_model):
    normalized_embs = []
    if args.norm_option == "standard":
        for word in range(embeddings.shape[0]):
            sample_embeds = randomSample_from_embeddings(-1, "en").to(device)
            std, mean = torch.std_mean(input=sample_embeds, dim=0, unbiased=True)    
            normalized_embs.append((embeddings[word,:] - mean) / std)
        normalized_embs = torch.stack(normalized_embs, dim=0)
        
    elif args.norm_option == "mapper-disc" or args.norm_option == "mapper-lossy":
        mapperLayer = load_mapper(vae_model, args.norm_option)
        with torch.no_grad():
            normalized_embs = mapperLayer(embeddings).to(device)
    elif args.norm_option == "none":
        normalized_embs = embeddings

    return normalized_embs

def interpolate_me_baby(emb1,emb2,n):
    """
      a diabolical function that linearly interpolates two tensors
      n: you better have this high
    """
    
    shapes = emb1.shape
    emb1_flattened_n = emb1.flatten().cpu().numpy()
    emb2_flattened_n = emb2.flatten().cpu().numpy()
    f = interp1d(x=[0,1], y=np.vstack([emb1_flattened_n,emb2_flattened_n]),axis=0)
    y = f(np.linspace(0, 1, n))
    L = [torch.reshape(torch.from_numpy(kral).float(), shapes).to(device) for kral in y]
    return torch.stack(L, dim=0) # return shape [n, latent_dim]

def add_text_to_image(tensir, word): #tensir: tenSÖR DID YOU GOT THE JOKE :d
    width, height = tensir.shape
    img_pil = transforms.functional.to_pil_image(tensir) #.resize((width,height + 30),0) #resmin altına yazı için boşluk
    draw = ImageDraw.Draw(img_pil)
    fontsize =  40
    font = ImageFont.truetype("./TimesNewRoman400.ttf", size=fontsize)  #font olayını ayarlayamadım, fontu truetype dosyası olarak indirmek gerekiyor
    wordSize = font.getsize(word)
    #draw.text(xy = (width//3,height+15), text = word, fill = "white",font = font, anchor = "ms") #fontu ayarlamayınca da şu anchor'lama olayı yapılamıyormuş. yazı şu anda sol altta, ben alt ortada olsun istemiştim ama böyle de güzel oldu.
    draw.text(xy = (int(width/2)-int(wordSize[0]/2),height-wordSize[1]-5), text = word, fill = "black", font = font)
    processed_tensir = to_tensor(img_pil)
    return torch.reshape(processed_tensir, [width,height])

def make_transparent(tensor_img):
    img = transforms.functional.to_pil_image(tensor_img)
    img = img.convert("RGBA")
    datas = img.getdata()

    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    img.putdata(newData)
    return to_tensor(img)


def savegrid(ims, rows=None, cols=None, fill=True, showax=False):
    if rows is None != cols is None:
        raise ValueError("Set either both rows and cols or neither.")

    if rows is None:
        rows = len(ims)
        cols = 1

    gridspec_kw = {'wspace': 0, 'hspace': 0} if fill else {}
    fig,axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw)

    if fill:
        bleed = 0
        fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax,im in zip(axarr.ravel(), ims):
        ax.imshow(im, cmap='gray')
        if not showax:
            ax.set_axis_off()

    kwargs = {'pad_inches': .01} if fill else {}
    fig.savefig(args.export_path+'logograms.png', **kwargs)

def post_proccess(imgs):
    imgs = list(map(lambda x:resize(to_pil_image(x),256), imgs))
    imgs = list(map(lambda x:adjust_saturation(x,1.5), imgs))
    imgs = list(map(lambda x:adjust_gamma(x,1.5), imgs))
    imgs = list(map(lambda x:adjust_contrast(x,2), imgs))
    imgs = list(map(lambda x:to_tensor(x), imgs))
    for x in imgs:
        x[x>0.5]=1
    for x in imgs:
        x[x<0.4]=0

    #save_image(imgs, 'img3.png')
    return imgs[0]

def main():
    print("started...")    

    vae_model = load_vae_model(args.vae_ckp_path).to(device)

    word_embeddings = embed_words()

    normalized_embeddings = normalize_embeddings(word_embeddings, vae_model)

    if args.siamese_mapper == "True":
        siam_mapper = load_mapper(vae_model, mapper_type="siamese_mapper")
        with torch.no_grad():
            normalized_embeddings = siam_mapper(normalized_embeddings).to(device)
        normalized_embeddings = normalize_embeddings(normalized_embeddings, vae_model)


    if args.gif == "True":
        n = 5
        interpolations = []
        for word in range(normalized_embeddings.shape[0]-1):
            w1 = normalized_embeddings[word,:128]
            w2 = normalized_embeddings[word+1,:128]
            
            intplt = interpolate_me_baby(w1, w2, n)
            interpolations.append(intplt)
        interpolations = torch.cat(interpolations, dim=0).to(device)

        images = []
        for word in range(interpolations.shape[0]):
            if word % n == 0:
                word_duration = 10
                textWord = args.words[int(word/n)] #.split(" - ")[0]
            else:
                word_duration = 1
                textWord = ""
            
            for duration in range(word_duration):
                logogram = vae_model.decode(interpolations[word,:128].float())
                # logogram = torch.cat([torch.zeros(1,1,64,64).float(), logogram.data.cpu()], dim=1).cpu()
                # images.append(logogram)
                processed = post_proccess(torch.reshape(logogram.data, [1,64,64]).cpu())
                processed = torch.reshape(processed, [256,256])
                processed = add_text_to_image(processed, textWord)
                images.append(processed)
        imageio.mimwrite(args.export_path+"helloToWorld.gif", images)

    else:
        images = []
        for word in range(normalized_embeddings.shape[0]):
            logogram = vae_model.decode(normalized_embeddings[word,:128].float())
            processed = post_proccess(torch.reshape(logogram.data, [1,64,64]).cpu())
            processed = torch.reshape(processed, [256,256])
            processed = add_text_to_image(processed, args.words[word])
            images.append(processed)
        savegrid(images, cols=int(math.sqrt(len(args.words))), rows=int(len(args.words)/int(math.sqrt(len(args.words)))))




        # vutils.save_image(logogram.data,
        #             args.export_path+args.words[word]+".png",
        #             normalize=True,
        #             nrow=12)



    print("ALL DONE!")


# python logogram_language_generator.py --embd_random_samples ./randomSamples_fromEmbeddings/sample_fasttext_embs_en_128.pickle --vae_ckp_path ./model_checkpoints/final_model_checkpoint.ckpt  --config_path ./configs/bbvae_CompleteRun.yaml --export_path ./outputs/ --umwe_mappers_path ./model_checkpoints/ --words_path words.txt --norm_option standard --mapper_layer_ckp ./vae_with_disc/mapper_checkpoint.pt --emb_bins_path ./model_checkpoints/cc_bins_128/ --gif False


if __name__ == "__main__":
    main()

import yaml
import argparse
import numpy as np

from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger

from torchvision.utils import save_image
from umwe2vae import umwe2vae


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()


def randomSample_from_embeddings(batch_size):
    with open(args.embd_random_samples, "rb") as f:
        random_sample = pickle.load(f)

    fake_batch = random.sample(list(random_sample.values()), batch_size)
    fake_batch = torch.stack(fake_batch, dim=0)
    return fake_batch


with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

model = vae_models[config['model_params']['name']](**config['model_params'])

checkpoint = torch.load("./model_checkpoints/final_model_checkpoint.ckpt")
state_dict = {}
for k in checkpoint['state_dict'].keys():
    state_dict[k[6:]] = checkpoint['state_dict'][k]

model.load_state_dict(state_dict)

ldr = [torch.randn((64,128)) for _ in range(10)]

inp = torch.randn((5,128))
u2v = umwe2vae(model, 128, 128)
save_image([u2v(inp)[0],u2v(inp)[1],u2v(inp)[2],u2v(inp)[3],u2v(inp)[4]], 'img1.png')
u2v.train(ldr)

save_image([u2v(inp)[0],u2v(inp)[1],u2v(inp)[2],u2v(inp)[3],u2v(inp)[4]], 'img2.png')

checkpoint = torch.load("./umwe2vae.ckpt")
u2v.load_state_dict(checkpoint['model_state_dict'])

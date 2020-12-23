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

# import fasttext
# import fasttext.util




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


def randomSample_from_embeddings(batch_size, fr=""):
    with open("/content/drive/My Drive/vae/logogram-language-generator-master/sample_fasttext_embs"+fr+".pickle", "rb") as f:
        random_sample = pickle.load(f)

    fake_batch = random.sample(list(random_sample.values()), batch_size)
    fake_batch = torch.stack(fake_batch, dim=0)
    return fake_batch
    




def trainer(vae_model, mapper, netD, batch_size, device):
    # Training Loop
    # Lists to keep track of progress
    img_list = []
    dataloader = get_dataLoader(batch_size=batch_size)

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = randomSample_from_embeddings(batch_size).to(device)




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
                randomsamples = []
                # for i in range(64):
                
                with torch.no_grad():
                  mapped = mapper(torch.randn(64,300).to(device))
                randomsamples.append(mapped.to(device))

                embed_samples = []
                # for i in range(64):
                one_embed = randomSample_from_embeddings(64).to(device)
                with torch.no_grad():
                  mapped_embed = mapper(one_embed).to(device)
                embed_samples.append(mapped_embed)
                


                randomsamples = torch.stack(randomsamples, dim=0)
                recons_randoms = vae_model.decode(randomsamples)

                embedsamples = torch.stack(embed_samples, dim=0)
                recons_embeds = vae_model.decode(embedsamples)



                vutils.save_image(recons_randoms.data,
                              f"./vae_with_disc/test_random{i}.png",
                              normalize=True,
                              nrow=12)
                vutils.save_image(recons_embeds.data,
                              f"./vae_with_disc/test_embed{i}.png",
                              normalize=True,
                              nrow=12)

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = mapper(fixed_noise).detach().to(device)
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                # vutils.save_image(img_list,
                #               f"./vae_with_disc/samples_{i}.png",
                #               normalize=True,
                #               nrow=12)

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
    

    with open("./configs/bbvae_setup2.yaml", 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    vae_checkpointPath = "logs/BetaVAE_B_setup2_run2/final_model_checkpoint.ckpt"
    batch_size = 64
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(device)
    
    vae_model = load_vae_model(vae_checkpointPath, config).to(device)

    # fixed_noise = randomSample_from_embeddings(batch_size).to(device)
    # fixed_noise = torch.randn()

    best_mapping = "./best_mapping_fr2en.t7"
    Readed_t7 = torch.from_numpy(torch.load(best_mapping)).to(device)

    with open("/content/drive/My Drive/vae/logogram-language-generator-master/fasttext_hello_world_fr.pickle", "rb") as f:
      helloworld = pickle.load(f)

    # helloworld = random.sample(list(helloworld.values()), batch_size)
    hello = torch.from_numpy(helloworld["hello"]).to(device)
    world = torch.from_numpy(helloworld["world"]).to(device)
    # helloworld = torch.stack([hello, world], dim=0).to(device)
    
    print(type(hello))
    print(type(Readed_t7))
    mapped_hello = torch.matmul(hello, Readed_t7)
    mapped_world = torch.matmul(world, Readed_t7)
    



    sample_embeds = randomSample_from_embeddings(batch_size, "_fr").to(device)

    std, mean = torch.std_mean(input=sample_embeds, dim=0, unbiased=True)
    # norm_func = transforms.Normalize(mean, std)
    ## norm here
    # normalized = norm_func(hello)
    
    normalized_hello = (mapped_hello - mean) / std
    normalized_world = (mapped_world - mean) / std
    
    recons_embeds = vae_model.decode(normalized_hello[:128])
    vutils.save_image(recons_embeds.data,
                  "./vae_with_disc/hello_normalized2_fr.png",
                  normalize=True,
                  nrow=12)
    recons_embeds = vae_model.decode(normalized_world[:128])
    vutils.save_image(recons_embeds.data,
                  "./vae_with_disc/World_normalized2_fr.png",
                  normalize=True,
                  nrow=12)



    print("ALL DONE!")





if __name__ == "__main__":
    main()
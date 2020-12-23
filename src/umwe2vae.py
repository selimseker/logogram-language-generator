import torch
from torch import nn
import torch.optim as optim
from torchvision.transforms.functional import adjust_contrast

class umwe2vae(nn.Module):
    def __init__(self, vae_model, in_dim=300, out_dim=128):
        super(umwe2vae, self).__init__()
        self.vae_model = vae_model
        #self.fc = nn.Linear(in_dim, out_dim)
        self.mapper_numlayer = 3

        self.linear_layers = []
        self.batch_norms = []
        for layer in range(0, self.mapper_numlayer-1):
            self.linear_layers.append(nn.Linear(in_dim, in_dim))
            self.batch_norms.append(nn.BatchNorm1d(in_dim))

        # final layer
        self.linear_layers.append(nn.Linear(in_dim, out_dim))
        self.batch_norms.append(nn.BatchNorm1d(out_dim))


        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.batch_norms = nn.ModuleList(self.batch_norms)


        self.relu = nn.ReLU()




    def forward(self, x):
        inp = x
        for layer in range(self.mapper_numlayer):
            out = self.linear_layers[layer](inp)
            out = self.batch_norms[layer](out)
            out = self.relu(out)
            inp = out        
        return out
        
        # h = self.fc(x)
        # y = self.vae_model.decode(h)
        # return h
        # here used to live post-processing
        #out = torch.zeros(y.shape)
        #for i in range(y.shape[0]):
        #    out[i] = adjust_contrast(y[i], contrast_factor=2.5)
        #return out

    def loss(self, x, alpha=1, beta=1):
        middle = x[:,:,1:-1,1:-1]
        ne     = x[:,:,0:-2,0:-2]
        n      = x[:,:,0:-2,1:-1]
        nw     = x[:,:,0:-2,2:]
        e      = x[:,:,1:-1,0:-2]
        w      = x[:,:,1:-1,2:]
        se     = x[:,:,2:,0:-2]
        s      = x[:,:,2:,1:-1]
        sw     = x[:,:,2:,2:]

        return alpha * torch.mean(sum([torch.abs(middle-ne),torch.abs(middle-n),torch.abs(middle-nw),torch.abs(middle-e),torch.abs(middle-w),torch.abs(middle-se),torch.abs(middle-s),torch.abs(middle-sw)]) / 8.) - beta * torch.mean(torch.abs(x-0.5))

    def train(self, loader, lr=0.001, epochs=5):
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0
            for _,inp in enumerate(loader):
                optimizer.zero_grad()

                out = self.forward(inp)
                out = self.vae_model.decode(out)
                loss = self.loss(out)

                loss.backward()
                optimizer.step()

                total_loss += loss

            print("Epoch: %d Loss: %f" % (epoch+1, total_loss/len(loader)))


        torch.save(self.state_dict(), "umwe2vae.ckpt")


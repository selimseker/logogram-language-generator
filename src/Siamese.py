import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

    def train_triplet(self, loader, epochs=100):
        self.train()
        print("new_siamese")
        optimizer = optim.Adam(self.parameters(), lr=0.00001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        minLoss = 1000.5
        patience = 0
        maxPatience = 5
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
            
            if total_loss.item()/len(loader) < minLoss:
                minLoss = total_loss.item()/len(loader)
                patience = 0
            else:
                patience +=1
                print("patienceCounter: ", patience)
                if patience == maxPatience:
                    print("early stopping: ", minLoss)
                    break
        torch.save(self.state_dict(), "siamese.ckpt")
        #torch.save({'model_state_dict': self.state_dict()}, "siamese_EPOCH100.ckpt")

    def eval_triplet(self, loader):
        self.eval()
        correct,total = 0,0
        for i,(xa,xp,xn) in enumerate(loader):
            yp = self.forward(xa.cuda(), xp.cuda())
            yn = self.forward(xa.cuda(), xn.cuda())
            correct += torch.sum((yp>0).float()) + torch.sum((yn<0).float()) 
            #correct += torch.sum(torch.FloatTensor(yp>0)) + torch.sum(torch.FloatTensor(yn<0))
            #correct += torch.sum(yp>0) + torch.sum(yn<0)
            total += yp.shape[0] + yn.shape[0]

            if i==100:
                break

        print("Accuracy: %f" % (correct/total).item())


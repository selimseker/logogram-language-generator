from Omniglot_triplet import Omniglot_triplet
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from Siamese import Siamese
import torch

#omni = Omniglot_triplet()
#dloader = DataLoader(omni,batch_size=16,shuffle=True)

siam = Siamese().cuda()
#siam.train_triplet(dloader)
siam.load_state_dict(torch.load("siamese.ckpt"))
siam.eval()
omnieval = Omniglot_triplet(split="test")
dloader = DataLoader(omnieval,batch_size=16,shuffle=True)
siam.eval_triplet(dloader)

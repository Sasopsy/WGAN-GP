import torch
import torch.nn as nn
from dataclasses import dataclass
import utils


@dataclass
class ModelArgs:
    image_shape: int = 64
    leaky_relu_alpha: float = 0.2
    latent_vector_shape: int = 128
    features_d: int = 64
    features_g: int = 64
    

class Reshape(nn.Module):
    
    def __init__(self,out_shape: tuple[int]):
        super(Reshape,self).__init__()
        self.reshape = torch.reshape
        self.out_shape = out_shape
        
    def forward(self,x):
        return self.reshape(x,shape=(x.shape[0],)+self.out_shape)


class Critic(nn.Module):
    
    def __init__(self,
                 args: ModelArgs):
        super(Critic, self).__init__()
        self.args = args
        self.disc = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=self.args.features_d,\
                kernel_size=4,stride=2,padding=1),  
            self.block(self.args.features_d,2*self.args.features_d,kernel_size=4,stride=2,padding=1),  
            self.block(2*self.args.features_d,4*self.args.features_d,kernel_size=4,stride=2,padding=1),  
            self.block(4*self.args.features_d,8*self.args.features_d,kernel_size=4,stride=2,padding=1), 
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(8*self.args.features_d, 1, kernel_size=4, stride=2, padding=0),
        )
        
    def block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(self.args.leaky_relu_alpha),
        )
        
    def forward(self,x):
        return self.disc(x)
    
    
class Generator(nn.Module):
    
    def __init__(self,
                 args: ModelArgs):
        super(Generator,self).__init__()
        self.args = args
        self.gen = nn.Sequential(
            self.block(self.args.latent_vector_shape,16*self.args.features_g, kernel_size=4, stride=1, padding=0),
            self.block(16*self.args.features_g,8*self.args.features_g,kernel_size=4,stride=2,padding=1),
            self.block(8*self.args.features_g,4*self.args.features_g,kernel_size=4,stride=2,padding=1),
            self.block(4*self.args.features_g,2*self.args.features_g,kernel_size=4,stride=2,padding=1),
            nn.ConvTranspose2d(2*self.args.features_g,3,kernel_size=4,stride=2,padding=1),
            nn.Sigmoid(),
        )
        
    def block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        
    def forward(self,x):
        return self.gen(x)
    
    def generate(self,device='cpu'):
        self.eval()
        latent_vec = torch.randn((1,self.args.latent_vector_shape,1,1)).to(device)
        self.train()
        return self(latent_vec)
    
        
def initialize_weights(model):
    # Initializes weights according to the DCGAN paper.
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            
def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 128
    x = torch.randn((N, in_channels, H, W))
    disc = Critic(ModelArgs())
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(ModelArgs())
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
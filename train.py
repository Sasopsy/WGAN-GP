import torch
import torch.nn as nn
from dataclasses import dataclass
import utils
import model
import dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import os
import time
from PIL import Image



@dataclass
class TrainConfig:
    root_directory: str = 'landscapes'
    save_directory: str = 'train_instance_1'
    image_shape: tuple = (64,64)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # Training Hyperparameters
    lambda_gp: float = 10.0
    critic_iterations: int = 1
    batch_size: int = 32
    learning_rate: float = 3e-4
    epochs: int = 1  # How many epochs to train s
    shuffle: bool = True
    # Misc
    print_every: int = 5  # Gap between printing results.
    cols: int = 8  # Number of columns in image subplot.
    save_every: int = 20 # Save every save_every iterations model checkpointing

    def __post_init__(self):
        self.model_args = model.ModelArgs(image_shape=self.image_shape[0])  # Overriding model args image shape.
        # Initialize generator and critic and load them to device.
        self.gen = model.Generator(self.model_args).to(self.device)
        self.critic = model.Critic(self.model_args).to(self.device)
        # Initlialise parameters according to DCGAN paper
        model.initialize_weights(self.critic)
        model.initialize_weights(self.gen)
        # Initialise optimizers
        self.opt_gen = optim.Adam(self.gen.parameters(),lr=self.learning_rate)
        self.opt_critic = optim.Adam(self.critic.parameters(),lr=self.learning_rate)


class TrainerWGAN():
    
    def __init__(self,
                 train_configs: TrainConfig):
        self.train_configs = train_configs
        
        self.gen = self.train_configs.gen
        self.critic = self.train_configs.critic
        self.opt_critic = self.train_configs.opt_critic
        self.opt_gen = self.train_configs.opt_gen
        
        # Get list of subdirectories in root directory.
        self.datasets = os.listdir(self.train_configs.root_directory)
        if '.DS_Store' in self.datasets:  # For MacOS.
            self.datasets.remove('.DS_Store') 
        self.datasets.sort()
        
        # Loss tracking
        self.history = []
        
        # Make save directory.
        os.mkdir(self.train_configs.save_directory)
        
        # Create checkpoint counter
        self.checkpoint_counter = 0
        

    def calculate_critic_loss(self,critic_real,critic_fake,gp):
        """Calculates # -(E[f(x_real)] - E[f(x_fake)] + lambda*gp)"""
        loss = -(torch.mean(critic_real) - torch.mean(critic_fake) + self.train_configs.lambda_gp*gp)
        return loss
        
    def calculate_gen_loss(self,critic_fake):
        """Calculates -E[f(x_fake)]"""
        loss = -(torch.mean(critic_fake))
        return loss
    
    def calculate_gradient_penalty(self,real_image_batch,fake_images):
        """Calculates (||/nabla_{x_cap}D|| - 1)**2 """
        B,C,H,W = real_image_batch.shape
        # Generate noise.
        alpha = torch.rand((B,1,1,1)).repeat((1,C,H,W)).to(self.train_configs.device)
        
        # Calculate interpolation. (x_cap)
        interpolated_images = real_image_batch*alpha + (1-alpha)*fake_images
        
        # Calculate critic score.
        critic_score = self.critic(interpolated_images)
        
        # Calculate gradient score.
        grad = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=critic_score,
            grad_outputs=torch.ones_like(critic_score),
            retain_graph=True,
            create_graph=True
        )[0]  # It returns a tuple of gradients. Hence we take the 0'th element.
        
        grad = grad.reshape(B,-1)
        grad_norm = grad.norm(2,dim=1)  # Calculate L2 norm across dimension 1. 0 is the batch dimension.
        gp = torch.mean((grad_norm-1)**2)
        
        return gp
    
    def train_step(self,real_image_batch):
        # Send real images to device.
        real_image_batch = real_image_batch.to(self.train_configs.device)
        
        # Train Discriminator.
        for _ in range(self.train_configs.critic_iterations):
            # Put critic to train and generator to eval mode.
            self.critic.train()
            self.gen.eval()
            # Sample random points from prior distribution (Here normal distribution).
            latent_vecs = torch.randn((real_image_batch.shape[0],\
                self.train_configs.model_args.latent_vector_shape,1,1)).to(self.train_configs.device)
            # Generate fake images.
            fake_images = self.gen(latent_vecs)
            
            # Calculate critic scores for real and fake images.
            critic_real = self.critic(real_image_batch).reshape(-1)
            critic_fake = self.critic(fake_images.detach()).reshape(-1)  # Will detach fake_images from graph
            
            # Calculate gradient penalty
            gp = self.calculate_gradient_penalty(real_image_batch,fake_images)
            
            # Calculate loss and optimize critic
            loss_critic = self.calculate_critic_loss(critic_real,critic_fake,gp)
            self.critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            self.opt_critic.step()
        
        # Train Generator
        # Put critic to eval and generator to train mode.
        self.critic.eval()
        self.gen.train()
        critic_fake = self.critic(fake_images).reshape(-1)
        loss_gen = self.calculate_gen_loss(critic_fake)
        self.gen.zero_grad()
        loss_gen.backward()
        self.opt_gen.step()
        
        # Update history object
        self.history.append(loss_critic.mean().item())
        
        # Store dictionary of mean of all the recent losses and fake images.
        self._dict = {
            "critic_loss": loss_critic.mean().item(),
            "gen_loss": loss_gen.mean().item(),
            "images": fake_images.to('cpu').permute(0, 2, 3, 1).detach().numpy(),
        }
        
    def train(self):
        
        # Create dataset instance.
        _dataset = dataset.Landscape(self.train_configs)
        
        self.dataloader = DataLoader(_dataset,
                                batch_size=self.train_configs.batch_size,
                                shuffle=self.train_configs.shuffle)
        
        for epoch in range(self.train_configs.epochs):
            
            loop = tqdm(enumerate(self.dataloader))
            
            for batch_idx,real_image_batch in loop:
                self.train_step(real_image_batch)
                
                # Print results
                if batch_idx%self.train_configs.print_every == 0:
                    print(f"Critic Loss: {self._dict['critic_loss']:.4f} | Gen Loss: {self._dict['gen_loss']:4f}")
                    # Print fake images.
                    utils.view_batch(self._dict['images'],self.train_configs.cols)
                
                # Save models.
                if batch_idx%self.train_configs.save_every == 0:
                    self.save(epoch=epoch,iteration=batch_idx)  
                
                # Set postfix in tqdm
                loop.set_postfix(loss=self._dict['critic_loss'])
        
    def save(self,epoch=None,iteration=None):
        """Saves train file and the results of the latest models' results."""
        
        # Create directory
        if epoch!=None and iteration!=None:
            dir_path = os.path.join(self.train_configs.save_directory,\
                f'epoch_{epoch}_{iteration}')
            os.mkdir(dir_path)
        else:
            dir_path = os.path.join(self.train_configs.save_directory,f'checkpoint_{self.checkpoint_counter}')
        
        save_file = os.path.join(dir_path,\
            f'train_file.pt')
        
        # Dump the train object in the sub-directory.
        torch.save({
            'critic_state_dict': self.critic.state_dict(),
            'gen_state_dict': self.gen.state_dict(),
            'opt_critic_state_dict': self.opt_critic.state_dict(),
            'opt_gen_state_dict': self.opt_gen.state_dict(),
            'train_configs': self.train_configs,
            'history': self.history
        },save_file)
            
        # Save images in separate directory.
        images = self._dict['images']*255
        images_path = os.path.join(dir_path,'images')
        os.mkdir(images_path)
        
        # Save each image individually.
        for i in range(images.shape[0]):
            # Convert NumPy array to Pillow Image.
            image = Image.fromarray(images[i].astype('uint8'))
            # Get path.
            path = os.path.join(images_path,f'image_{i}.png')
            # Save the image.
            image.save(path)
        
        self.checkpoint_counter+=1

    @classmethod
    def load(cls,
             save_directory,
             particular_instance: str = None):
        """Returns the latest models trained if particular_instance is none."""
        """particular_instance format: epoch_(num_epoch)_iteration_(num_iteration) or checkpoint_(num)"""
        _list = os.listdir(save_directory)
        _list.sort()
        if '.DS_Store' in _list:  # For MacOS.
            _list.remove('.DS_Store') 
            
        if particular_instance:
            path = os.path.join(save_directory,particular_instance,f'train_file.pt')
        else:
            path_dir = os.path.join(save_directory,_list[-1])
            _sublist = os.listdir(path_dir)  
            _sublist.sort()
            # Select the final file in the directory as it starts with 't'. The rest
            # starts with alphabets preceding it. (i, and D)
            path = os.path.join(path_dir,_sublist[-1])
        
        # Dummy config.
        dummy_config = TrainConfig()
        
        # Create class.
        self = cls(dummy_config)
        
        # Load the state.
        load_dict = torch.load(path)
        
        # Load the necessary attributes.
        self.history = load_dict['history']
        self.train_configs = load_dict['train_configs']
        self.critic.load_state_dict(load_dict['critic_state_dict'])
        self.gen.load_state_dict(load_dict['gen_state_dict'])
        self.opt_critic.load_state_dict(load_dict['opt_critic_state_dict'])
        self.opt_gen.load_state_dict(load_dict['opt_gen_state_dict'])
        
        # Delete dummy config.
        del dummy_config
        
        return self
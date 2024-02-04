import config
from config import load_weights, save_weights, initialize_conv_weights_normal
from dataset import PhotoDataset, transform, denorm
from generator import Generator, train_generator
from discriminator import Discriminator, train_disc
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import torch
import os

def fit():
    generator_A = Generator().to(config.DEVICE)
    generator_B = Generator().to(config.DEVICE)
    
    discriminator_A = Discriminator().to(config.DEVICE)
    discriminator_B = Discriminator().to(config.DEVICE)

    disc_opt = optim.Adam(list(discriminator_A.parameters()) + list(discriminator_B.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999))
    gen_opt = optim.Adam(list(generator_A.parameters()) + list(generator_B.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999))

    if config.LOAD_MODEL:
        load_weights(config.WEIGHTS_GEN_A, generator_A, gen_opt, config.LEARNING_RATE)
        load_weights(config.WEIGHTS_GEN_B, generator_B, gen_opt, config.LEARNING_RATE)
        load_weights(config.WEIGHTS_DISC_A, discriminator_A, disc_opt, config.LEARNING_RATE)
        load_weights(config.WEIGHTS_DISC_B, discriminator_B, disc_opt, config.LEARNING_RATE)
    else:
        initialize_conv_weights_normal(generator_A)
        initialize_conv_weights_normal(generator_B)
        initialize_conv_weights_normal(discriminator_A)
        initialize_conv_weights_normal(discriminator_B)

    Train_dataset = PhotoDataset(transform=transform, root_A='trainA', root_B='trainB')
    Train_dataloader = DataLoader(Train_dataset, batch_size=1, shuffle=True, pin_memory=True)
    
    disc_loss_list = []
    gen_loss_list = []
    
    sample_dir = 'training_samples'
    os.makedirs(sample_dir, exist_ok=True)
    
    for epoch in range(config.NUM_EPOCHS):
        for idx, (image_A, image_B) in enumerate(tqdm(Train_dataloader)):
            image_A = image_A.to(config.DEVICE)
            image_B = image_B.to(config.DEVICE)
            fake_A, fake_B, disc_loss = train_disc(image_A, image_B, discriminator_A, discriminator_B, generator_A, generator_B, disc_opt)
            disc_loss_list.append(disc_loss)
            gen_loss = train_generator(image_A, image_B, fake_A, fake_B, discriminator_A, discriminator_B, generator_A, generator_B, gen_opt)
            gen_loss_list.append(gen_loss)
            if idx % 100 == 0:
                save_image(denorm(fake_A), f"{sample_dir}/B2A_epoch_{epoch+1}_{idx}.png")
                save_image(denorm(image_B), f"{sample_dir}/ORIG_epoch_{epoch+1}_{idx}.png")  

        if config.SAVE_MODEL:
            save_weights(generator_A, gen_opt, filename=config.WEIGHTS_GEN_A)
            save_weights(generator_B, gen_opt, filename=config.WEIGHTS_GEN_B)
            save_weights(discriminator_A, disc_opt, filename=config.WEIGHTS_DISC_A)
            save_weights(discriminator_B, disc_opt, filename=config.WEIGHTS_DISC_B)

    return disc_loss_list, gen_loss_list


def start():
    torch.cuda.empty_cache()
    fit()

if __name__ == "__main__":
    start()
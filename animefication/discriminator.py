import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminate = nn.Sequential(
            # in: 3 x 256 x 256
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            # We do not use InstanceNorm for the first C64 layer
            nn.LeakyReLU(0.2, inplace=True),
        
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"),  
            # out: 1 x 30 x 30 (patches)
            
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.discriminate(x)
        return x

def train_disc(image_A, image_B, disc_A, disc_B, gen_A, gen_B, optimizer):
    optimizer.zero_grad() # nullifying gradients
    # discriminator A train
    fake_A = gen_B(image_B) # generate picture of A domain with second generator from B picture
    disc_A_real_result = disc_A(image_A) # getting result of discriminator for A domain for real(A) picture
    disc_A_fake_result = disc_A(fake_A.detach()) # getting result of discriminator for A domain for fake(A) picture that we generate 2 steps earlier
    # detach need to avoid second gradient calculation when we train generator
    mse = nn.MSELoss() 
    disc_A_real_loss = mse(disc_A_real_result, torch.ones_like(disc_A_real_result)) # MSE loss for real image 
    disc_A_fake_loss = mse(disc_A_fake_result, torch.zeros_like(disc_A_fake_result)) # MSE loss for fake image
    disc_A_loss = disc_A_fake_loss + disc_A_real_loss # total diacriminator A loss
    # discriminator B train
    fake_B = gen_A(image_A) # generate picture of B domain with first generator from A picture
    disc_B_real_result = disc_B(image_B) # getting result of discriminator for B domain for real(B) picture
    disc_B_fake_result = disc_B(fake_B.detach()) # getting result of discriminator for B domain for fake(B) picture that we generate 2 steps earlier
    disc_B_real_loss = mse(disc_B_real_result, torch.ones_like(disc_B_real_result)) # MSE loss for real image (second domain)
    disc_B_fake_loss = mse(disc_B_fake_result, torch.zeros_like(disc_B_fake_result)) # MSE loss for fake image (second domain)
    disc_B_loss = disc_B_fake_loss + disc_B_real_loss # total diacriminator B loss

    disc_loss = disc_A_loss + disc_B_loss # total discriminators loss
    disc_loss.backward() # calculate gradients 
    optimizer.step() # make step
    return fake_A, fake_B, disc_loss.item()

def test():
    x = torch.randn((5, 3, 256, 256))
    model = Discriminator()
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test()
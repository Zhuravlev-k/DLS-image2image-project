import torch
import torch.nn as nn
import config
# residual block that contains two 3 × 3 convolutional layers with the same number of filters on both layer
class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.connect = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(filters),
            nn.Identity()
        )
    def forward(self, x):
        return x + self.connect(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # c7s1-k denote a 7×7 Convolution-InstanceNormReLU layer with k filters and stride 1.
        self.first = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # dk denotes a 3 × 3 Convolution-InstanceNorm-ReLU layer with k filters and stride 2. Reflection padding
        # d128,d256,
        self.down = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.residual = nn.Sequential(
            *[ResidualBlock(256) for _ in range(14)]
        )
        
        # uk denotes a 3 × 3 fractional-strided-ConvolutionInstanceNorm-ReLU layer with k filters and stride 1/2
        self.up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1,),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.last = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3, padding_mode="reflect")
        )

    def forward(self, x):
        x = self.first(x)
        x = self.down(x)
        x = self.residual(x)
        x = self.up(x)
        return torch.tanh(self.last(x))

def train_generator(image_A, image_B, fake_A, fake_B, disc_A, disc_B, gen_A, gen_B, optimizer):
    optimizer.zero_grad() # nullifying gradients
    # adversarial losses
   
    discriminator_A_fake = disc_A(fake_A)
    discriminator_B_fake = disc_A(fake_B)
    mse = nn.MSELoss()
    generator_A_loss = mse(discriminator_A_fake, torch.ones_like(discriminator_A_fake))
    generator_B_loss = mse(discriminator_B_fake, torch.ones_like(discriminator_B_fake))
    # cycle loss
    cycle_A = gen_A(fake_B)
    cycle_B = gen_B(fake_A)
    Lloss = nn.L1Loss()
    cycle_A_loss = Lloss(image_A, cycle_A)
    cycle_B_loss = Lloss(image_B, cycle_B)
    # identity loss 
    identity_A = gen_A(image_A)
    identity_B = gen_B(image_B)
    identity_A_loss = Lloss(image_A, identity_A)
    identity_B_loss = Lloss(image_B, identity_B)
    # total generator loss
    gen_loss = (generator_A_loss + generator_B_loss + (cycle_A_loss + cycle_B_loss) * config.LAMBDA_CYCLE + (identity_A_loss + identity_B_loss) * config.LAMBDA_IDENTITY)
    gen_loss.backward()
    optimizer.step()
    return gen_loss.item()


def test():
    x = torch.randn((1, 3, 256, 256))
    gen = Generator()
    print(gen(x).shape)


if __name__ == "__main__":
    test()
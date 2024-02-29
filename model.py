
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.norm(self.conv1(x)))
        out = self.norm(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1)
        self.norm = nn.InstanceNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(x4))
        return x1, x2, x3, x4, x5

class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.conv_query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv_key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        query = self.conv_query(x)
        key = self.conv_key(x)
        value = self.conv_value(x)
        attention_map = torch.matmul(query, key.transpose(-2, -1))
        attention_map = F.softmax(attention_map, dim=-1)
        out = torch.matmul(attention_map, value)
        return out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)
        self.norm = nn.InstanceNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.attention = Attention(1024)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.relu(self.norm(self.conv1(x5)))
        x = torch.cat((x, x4), dim=1)
        x = self.relu(self.norm(self.conv2(x)))
        x = torch.cat((x, x3), dim=1)
        x = self.relu(self.norm(self.conv3(x)))
        x = torch.cat((x, x2), dim=1)
        x = self.relu(self.norm(self.conv4(x)))
        x = torch.cat((x, x1), dim=1)
        x = self.tanh(self.conv5(x))
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.res_blocks = nn.Sequential(
            ResidualBlock(1024, 1024),
            ResidualBlock(1024, 1024),
            ResidualBlock(1024, 1024),
        )

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        x5 = self.res_blocks(x5)
        x = self.decoder(x1, x2, x3, x4, x5)
        return x


# Discriminator (PatchGAN)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


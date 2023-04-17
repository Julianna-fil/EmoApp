import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, c_dim):
        super(Generator, self).__init__()
        self.g = nn.Sequential(
            #-------Down-sampling--------------------
            nn.Conv2d(3+c_dim, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            #--------Bottleneck---------------------------

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            # (так 6 раз)
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),


            #-------Up-sampling-----------------------------
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh()
        )


    def forward(self, x, c):
        # labels = self.label_embedding(labels).view(-1, 1, self.config.noise_shape, self.config.noise_shape)
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        # print(f"size = {x.size()}")
        return self.g(x)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.d = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),

            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),

            nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01)
        )
        self.conv1 = nn.Conv2d(2048, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(2048, 2, kernel_size=4, bias=False)

    def forward(self, x):
        h = self.d(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        # print(out_cls.size())
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))


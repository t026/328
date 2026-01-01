class MyBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True, dropout_rate=0.1):
        super(MyBlock, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.GELU() if activation is None else activation
        self.normalize = normalize
        self.dropout = nn.Dropout(dropout_rate)
        
        # Add residual connection
        self.residual = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.activation(out)
        out = self.dropout(out)

        return out + identity


class UNet(nn.Module):
    def __init__(self, in_channels: int=1, 
                 down_channels =[64, 128, 128, 128, 128], 
                 up_channels=[128, 128, 128, 128, 64], 
                 time_emb_dim: int=128,
                 num_classes: int=10) -> None:
        super().__init__()

        # NOTE: You can change the arguments received by the UNet if you want, but keep the num_classes argument

        self.num_classes = num_classes

        # TODO: time embedding layer
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim*2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(time_emb_dim*2, time_emb_dim),
            nn.LayerNorm(time_emb_dim)  # Added layer normalization
        )

        # TODO: define the embedding layer to compute embeddings for the labels
        self.class_emb = nn.Embedding(num_classes, time_emb_dim)



        # define your network architecture here
        

        self.label_conditioning = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.GELU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        self.fc = nn.Sequential(nn.Linear(time_emb_dim*2, time_emb_dim), 
                                nn.Linear(time_emb_dim, 1))
        self.b1 = nn.Sequential(
            MyBlock((1, 32, 32), 1, 16),  
            MyBlock((16, 32, 32), 16, 16),
            MyBlock((16, 32, 32), 16, 16)
        )
        self.down1 = nn.Conv2d(16, 16, 4, 2, 1)

        self.b2 = nn.Sequential(
            MyBlock((16, 16, 16), 16, 32),  
            MyBlock((32, 16, 16), 32, 32),
            MyBlock((32, 16, 16), 32, 32)
        )
        self.down2 = nn.Conv2d(32, 32, 4, 2, 1)

        self.b3 = nn.Sequential(
            MyBlock((32, 8, 8), 32, 64),  
            MyBlock((64, 8, 8), 64, 64),
            MyBlock((64, 8, 8), 64, 64)
        )
        self.down3 = nn.Conv2d(64, 64, 4, 2, 1)

        self.b_mid = nn.Sequential(
            MyBlock((64, 4, 4), 64, 32),
            MyBlock((32, 4, 4), 32, 32),
            MyBlock((32, 4, 4), 32, 64),
            nn.Dropout(0.2) 
        )

        self.reduce = nn.Conv2d(128, 64, 1)
        self.up1 = nn.ConvTranspose2d(64, 64, 4, 2, 1)

        self.b4 = nn.Sequential(
            MyBlock((64, 8, 8), 64, 64),
            MyBlock((64, 8, 8), 64, 32),
            MyBlock((32, 8, 8), 32, 32)
        )

        self.up2 = nn.ConvTranspose2d(32, 32, 4, 2, 1)

        self.b5 = nn.Sequential(
            MyBlock((64, 16, 16), 64, 32),
            MyBlock((32, 16, 16), 32, 16),
            MyBlock((16, 16, 16), 16, 16)
        )

        self.up3 = nn.ConvTranspose2d(16, 16, 4, 2, 1)

        self.b_out = nn.Sequential(
            MyBlock((32, 32, 32), 32, 16),
            MyBlock((16, 32, 32), 16, 16),
            MyBlock((16, 32, 32), 16, 16, normalize=False)
        )

        self.conv_out = nn.Conv2d(16, 1, 3, 1, 1)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, label: torch.Tensor = None) -> torch.Tensor:
        # TODO: embed time
        t = self.time_mlp(timestep)

        # TODO: handle label embeddings if labels are available
        if label is not None:
            l = self.class_emb(label)
            label_cond = self.label_conditioning(l)
            t = torch.cat((t, label_cond), dim=1)
            
        
        # TODO: compute the output of your network
        n = len(x)
        t = self.fc(t).reshape(n, -1, 1, 1)
        
        out1 = self.b1(x + t.reshape(n, -1, 1, 1))  # (N, 16, 32, 32)
        out2 = self.b2(self.down1(out1) + t)  # (N, 32, 16, 16)
        out3 = self.b3(self.down2(out2) + t)  # (N, 64, 8, 8)

        out_mid = self.b_mid(self.down3(out3) + t)  # (N, 64, 4, 4)

        out4_c = torch.cat((out3, self.up1(out_mid)), dim=1)  # (N, 128, 8, 8)
        out4 = self.reduce(out4_c)
        out4 = self.b4(out4 + t)  # (N, 32, 8, 8)

        out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (N, 64, 16, 16)
        out5 = self.b5(out5 + t)  # (N, 16, 16, 16)

        out = torch.cat((out1, self.up3(out5)), dim=1)  # (N, 32, 32, 32)
        out = self.b_out(out + t)  # (N, 16, 32, 32)

        out = self.conv_out(out)

        return out
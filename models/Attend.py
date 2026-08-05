''' 
Code taken from https://github.com/AdelaideAuto-IDLab/Attend-And-Discriminate
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1d(ni: int, no: int, ks: int = 1, stride: int = 1, padding: int = 0, bias: bool = False):
    """
    Create and initialize a `nn.Conv1d` layer with spectral normalization.
    """
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias:
        conv.bias.data.zero_()
    return conv


class SelfAttention(nn.Module):
    """
    Self attention layer for nd
    """
    def __init__(self, n_channels: int, div):
        super(SelfAttention, self).__init__()

        if n_channels > 1:
            self.query = conv1d(n_channels, n_channels//div)
            self.key = conv1d(n_channels, n_channels//div)
        else:
            self.query = conv1d(n_channels, n_channels)
            self.key = conv1d(n_channels, n_channels)
        self.value = conv1d(n_channels, n_channels)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()


class TemporalAttention(nn.Module):
    """
    Temporal attention module
    """
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.fc = nn.Linear(hidden_dim, 1)
        self.sm = torch.nn.Softmax(dim=0)

    def forward(self, x):
        out = self.fc(x).squeeze(2)
        weights_att = self.sm(out).unsqueeze(2)
        context = torch.sum(weights_att * x, 0)
        return context


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        filter_num,
        filter_size,
        enc_num_layers,
        enc_is_bidirectional,
        dropout,
        dropout_rnn,
        activation,
        sa_div,
        t_context
    ):
        super(FeatureExtractor, self).__init__()

        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(1, filter_num, (filter_size, 1),padding=(1,0))
        self.conv2 = nn.Conv2d(filter_num, filter_num, (filter_size, 1),padding=(1,0))
        self.conv3 = nn.Conv2d(filter_num, filter_num, (filter_size, 1),padding=(1,0))
        self.conv4 = nn.Conv2d(filter_num, filter_num, (filter_size, 1))
        self.activation = nn.ReLU() if activation == "ReLU" else nn.Tanh()

        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(
            filter_num * input_dim,
            hidden_dim,
            enc_num_layers,
            bidirectional=enc_is_bidirectional,
            dropout=dropout_rnn,
        )

        self.ta = TemporalAttention(hidden_dim)
        self.sa = SelfAttention(filter_num, sa_div)

        # --- CONTINUOUS EXPONENTIAL DECAY SYSTEM ---
        self.num_sensors, self.window_size = t_context
        
        # Unique parameters per convolutional filter map and tracking sensor node
        # Layout matching shape: [1, filter_num, 1, num_sensors] for broadcasting
        self.decay_rate = nn.Parameter(torch.ones(1, filter_num, 1, self.num_sensors) * 0.1)
        self.decay_bias = nn.Parameter(torch.zeros(1, filter_num, 1, self.num_sensors))

    def forward(self, x, ages=None):
        x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        # x shape at this junction: [Batch, filter_num, window_size - 2, input_dim]
        if ages is not None:
            # 1. Expand raw ages from [B, num_sensors] to [B, 1, 1, num_sensors]
            age_grid = ages.unsqueeze(1).unsqueeze(2)
            
            # 2. Build continuous bounded exponential decay map [B, filter_num, 1, num_sensors]
            # Enforces safe monotonic decay between 0.0 and 1.0 via -ReLU
            decay_factor = torch.exp(-F.relu(self.decay_rate * age_grid + self.decay_bias))
            
            # 3. Handle coordinate duplication factor (Total input channels / physical sensor nodes)
            axes_per_sensor = x.shape[3] // self.num_sensors
            
            # 4. Replicate down the spatial axis dimension -> [B, filter_num, 1, input_dim]
            decay_grid = decay_factor.repeat_interleave(axes_per_sensor, dim=3)
            
            # 5. Continuous gating multiplication (automatically broadcasts across sequence timeline)
            x = x * decay_grid

        # apply self-attention on each temporal dimension (along sensor and feature dimensions)
        refined = torch.cat(
            [self.sa(torch.unsqueeze(x[:, :, t, :], dim=3)) for t in range(x.shape[2])],
            dim=-1,
        )
        x = refined.permute(3, 0, 1, 2)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.dropout(x)
        outputs, h = self.rnn(x)

        # apply temporal attention on GRU outputs
        out = self.ta(outputs)
        return out


class Classifier(nn.Module):
    def __init__(self, hidden_dim, num_class):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(hidden_dim, num_class)

    def forward(self, z):
        return self.fc(z)


class AttendDiscriminate(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        filter_num,
        filter_size,
        enc_num_layers,
        enc_is_bidirectional,
        dropout,
        dropout_rnn,
        dropout_cls,
        activation,
        sa_div,
        num_class,
        t_context = None
    ):
        super(AttendDiscriminate, self).__init__()

        self.fe = FeatureExtractor(
            input_dim,
            hidden_dim,
            filter_num,
            filter_size,
            enc_num_layers,
            enc_is_bidirectional,
            dropout,
            dropout_rnn,
            activation,
            sa_div,
            t_context
        )

        self.dropout = nn.Dropout(dropout_cls)
        self.classifier = Classifier(hidden_dim, num_class)

    def forward(self, x, ages=None):
        feature = self.fe(x, ages)
        z = feature.div(
            torch.norm(feature, p=2, dim=1, keepdim=True).expand_as(feature)
        )
        out = self.dropout(feature)
        logits = self.classifier(out)
        return logits


if __name__ == "__main__":
    import yaml
    from experiments.train_har_classifier import get_args
    
    config_file = open('model_configs.yaml', mode='r')
    config = yaml.load(config_file, Loader=yaml.FullLoader)["attend"]

    # synthetic HAR batch (batch size, window size, channels)
    args = get_args()
    args = vars(args)
    num_channels = 3*len(args['body_parts'])*len(args['sensors'])
    data_synthetic = torch.randn((args['batch_size'], args['window_size'], num_channels)).cuda()
    ages_synthetic = torch.randn((args['batch_size'], len(args['body_parts']))).cuda()

    # create HAR model
    model = AttendDiscriminate(input_dim=num_channels,**config,num_class=len(args['activities']),t_context=(len(args['body_parts']),args['window_size'])).cuda()
    model.eval()
    with torch.no_grad():
        logits = model(data_synthetic, ages_synthetic)
        print(f"\t input: {data_synthetic.shape} {data_synthetic.dtype}")
        print(f"\t logits: {logits.shape} {logits.dtype}")

        print(f"Num Params:{sum(p.numel() for p in model.parameters())}")

    for name,param in model.fe.named_parameters():
        print(name)
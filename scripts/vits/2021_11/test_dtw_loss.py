# Code in file nn/dynamic_net.py
import random
import torch
import modules
import commons

class FramePriorNet(torch.nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channels, kernel_size=5, n_layers=4) -> None:
        super().__init__()
        self.conv = torch.nn.ModuleList()
        for idx in range(n_layers):
            if idx != 0:
                in_channel = hidden_channel
            self.conv += [torch.nn.Sequential(
                torch.nn.Conv1d(in_channel, hidden_channel, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
                torch.nn.ReLU(),
                modules.LayerNorm(hidden_channel)
            )]
        self.proj = torch.nn.Conv1d(hidden_channel, out_channels * 2, 1)
        self.out_channels = out_channels

    def forward(self, x):
        for f in self.conv:
            x = f(x)
        stats = self.proj(x)
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
b, embed, length_min, length_max, length_target = 8, 192, 300, 600, 450
y_mask = torch.ones((b, length_target)).bool().cuda()
# Create random Tensors to hold inputs and outputs.
target_m = torch.ones((b, embed, length_target)).cuda()
target_logs = torch.ones((b, embed, length_target)).cuda()

model = FramePriorNet(embed, embed, embed).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

length_rand = int(torch.randint(length_min, length_max, size=(2,))[0])
input = torch.ones((b, embed, int(length_rand))).cuda()
    
from soft_dtw_loss import SoftDynamicTimeWarping
for i in range(1000):

    # Construct our model by instantiating the class defined above

    x, m, logs = model(input)
    m,logs = input, input
    # # Construct our loss function and an Optimizer. Training this strange model with
    # # vanilla stochastic gradient descent is tough, so we use momentum
    soft_dtw = SoftDynamicTimeWarping(penalty=1.0,gamma=0.01,bandwidth=120,dist_func="kl_divergence",average_alignment_save_path=".",device="cuda")
    x_mask = torch.ones((b, length_rand)).bool().cuda()

    # # Compute and print lossss
    loss = soft_dtw(torch.cat([m, logs],dim=1).transpose(1,2),
                    torch.cat([target_m, target_logs],dim=1).transpose(1,2),
                    x_mask,
                    y_mask)
    print(loss)
    # # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
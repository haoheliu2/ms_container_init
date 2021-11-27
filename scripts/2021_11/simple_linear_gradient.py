import torch
import torch.nn.functional as F 

torch.manual_seed(1234)

linear = torch.nn.Linear(10,10)

optimizer = torch.optim.SGD(linear.parameters(), lr=0.01, momentum=0.9)

a = torch.randn((4,10))
b = torch.randn((4,10))

a = linear(a)

optimizer.zero_grad()

loss1 = F.l1_loss(b,a)

loss1.backward()

for name, parms in linear.named_parameters():	
		print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
		 ' -->grad_value:',parms.grad)

optimizer.step()




# loss2 = F.l1_loss(b,a)

# print(loss1, loss2)
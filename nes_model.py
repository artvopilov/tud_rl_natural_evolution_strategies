import torch
from torch import nn
import torch.nn.functional as F


class ModelNes(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(ModelNes, self).__init__()

		self.linear1 = nn.Linear(input_dim, 32)
		self.linear2 = nn.Linear(32, 48)
		self.head = nn.Linear(48, output_dim)


	def forward(self, x):
		x = torch.relu(self.linear1(x))
		x = torch.relu(self.linear2(x))

		return self.head(x)

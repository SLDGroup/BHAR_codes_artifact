import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# per body part model to convert small windows to embeddings
class ConvEncoder(nn.Module):
	def __init__(self,in_channels,embd_dim):
		super().__init__()

		self.conv1 = nn.Conv1d(in_channels, 16, 3)
		self.conv2 = nn.Conv1d(16, 16, 3)
		self.conv3 = nn.Conv1d(16, embd_dim, 3)
	

	def forward(self, x):
		# (3 x 16) --> (16 x 14)
		x = F.relu(self.conv1(x))

		# (16 x 14) --> (16 x 12)
		x = F.relu(self.conv2(x))

		# (16 x 12) --> (embd_dim x 1)
		x = F.relu(self.conv3(x))

		# GAP
		x = torch.mean(x,dim=-1)

		# (embd_dim x 1) --> (embd_dim)
		x = x.view(x.shape[0],-1)
		return x

# self attention head
class SelfAttention(nn.Module):
	def __init__(self, head_size, embd_dim):
		super().__init__()
		self.head_size = head_size
		self.embd_dim = embd_dim
		self.query_embed = nn.Linear(embd_dim, head_size,bias=False)
		self.key_embed = nn.Linear(embd_dim, head_size,bias=False)
		self.value_embed = nn.Linear(embd_dim, head_size,bias=False)
		self.dropout = nn.Dropout(0.2)

	def forward(self, x,viz=False):
		B, L, D = x.shape
		k = self.key_embed(x) # (B, L, C)
		q = self.query_embed(x) # (B, L, C)

		wei = q @ k.transpose(-2,-1) * D**(-0.5)
		wei = F.softmax(wei,dim=-1)
		wei = self.dropout(wei)

		# if viz == True:
		self.wei = wei[0].to('cpu').mean(dim=0)

			# fig,ax = plt.subplots(1,1,figsize=(8,8))
			# ax.imshow(wei[0].to('cpu').mean(dim=0,keepdim=True))
			# fig.savefig('am.png')

		v = self.value_embed(x)
		out = wei @ v

		return out
	
class MultiHeadAttention(nn.Module):
	def __init__(self, num_heads, head_size, embd_dim):
		super().__init__()
		self.heads = nn.ModuleList([SelfAttention(head_size,embd_dim) for _ in range(num_heads)])
		self.proj = nn.Linear(embd_dim,embd_dim)
		self.dropout = nn.Dropout(0.2)

	def forward(self, x):
		out = torch.cat([h(x) for h in self.heads], dim=-1)
		out = self.dropout(self.proj(out))
		return out
	

class FeedForward(nn.Module):
	def __init__(self, embd_dim) :
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(embd_dim,2*embd_dim),
			nn.ReLU(),
			nn.Linear(2*embd_dim,embd_dim),
			nn.Dropout(0.2)
		)

	def forward(self, x):
		return self.net(x)
	

class Block(nn.Module):
	def __init__(self, embd_dim, num_heads) :
		super().__init__()
		head_size = embd_dim // num_heads
		self.sa = MultiHeadAttention(num_heads, head_size, embd_dim)
		self.ffwd = FeedForward(embd_dim)
		self.ln1 = nn.LayerNorm(embd_dim)
		self.ln2 = nn.LayerNorm(embd_dim)

	def forward(self, x):
		x = x + self.sa(self.ln1(x))
		x = x + self.ffwd(self.ln2(x))
		return x

# Transformer like model to analyze asynchronous segments
class AsyncFormer(nn.Module):
	def __init__(self,embd_dim,body_parts,n_classes):
		super().__init__()
		self.embd_dim = embd_dim
		self.body_parts = body_parts
		self.pos_embd = nn.Linear(2,embd_dim) # Input: (body part idx, relative arrival time)
		
		# a conv encoder for each body part
		self.conv_encoders = nn.ModuleList([ConvEncoder(3,embd_dim) for bp in range(body_parts)])

		self.blocks = nn.Sequential(
			Block(embd_dim, 4),
			Block(embd_dim, 4),
			# Block(embd_dim, 4),
			# Block(embd_dim, 4),
			nn.LayerNorm(embd_dim)
		)  

		self.fc = nn.Linear(embd_dim,n_classes) 
	

	def forward(self, window, arr_t):
		# window is (B x 7 x 5 x 16 x 3)
		# the effective batch size is B*5 for the conv encoder so need to review then reshape back
		B_w, K_w, L_w, T_w, C_w = window.shape 
		# print()
		# print('-------- Initial Input --------')
		# print(f'Window Shape: {window.shape}')
		window = window.permute(1,0,2,3,4).contiguous().view(K_w,B_w*L_w,T_w,C_w).transpose(-2,-1) # (K x B*L x C x T)
		# window = self.in_norm(window)

		B_a, K_a, L_a = arr_t.shape
		
		arr_t = arr_t.permute(1,0,2).reshape(K_a,B_a*L_a) # (K_a x B_a*L_a)
		
		bp_pos = torch.arange(self.body_parts).unsqueeze(1).expand(K_a, B_a*L_a).flatten().to('cuda')
		arr_t = torch.stack([arr_t.view(-1),bp_pos],-1) # (K_a*B_a*L_a x 2)

		# pass each batch through corresponding CNN encoder
		conv_embds = torch.stack([cv(win) for cv,win in zip(self.conv_encoders,window)]) # (K_w x B_w*L_w x D)
		
		pos_embds = self.pos_embd(arr_t).view(K_a,B_a*L_a,self.embd_dim) # (K_a x B_a*L_a x D)
		pos_embds = (pos_embds-pos_embds.mean())/pos_embds.std()

		# get a batch of embedding sequences
		embds = (conv_embds + pos_embds).reshape(K_a,B_a,L_a,self.embd_dim).permute(1,0,2,3).reshape(B_a,K_a*L_a,self.embd_dim) # (B x K*L x D)

		# self attention
		y = self.blocks(embds)
		# print()
		# print('-------- SA Output --------')
		# print(f'Embds Shape: {y.shape}')

		# GAP
		y = torch.mean(y,dim=1)
		# print()
		# print('-------- GAP --------')
		# print(f'Embds Shape: {y.shape}')

		# Classifier
		out = self.fc(y)
		# print()
		# print('-------- FC --------')
		# print(f'Out Shape: {out.shape}')
		return out
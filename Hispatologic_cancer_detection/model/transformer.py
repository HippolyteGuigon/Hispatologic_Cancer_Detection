import torch
import torch.nn as nn
import torchvision
from torch.utils.data import random_split
import sys
import os
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from Hispatologic_cancer_detection.configs.confs import *
from Hispatologic_cancer_detection.model_save_load.model_save_load import *
from Hispatologic_cancer_detection.transforms.transform import *
from Hispatologic_cancer_detection.metrics.metrics import *
from Hispatologic_cancer_detection.logs.logs import *
from Hispatologic_cancer_detection.early_stopping.early_stopping import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

tqdm.pandas()

main_params = load_conf("configs/main.yml", include=True)
main_params = clean_params(main_params)
batch_size = main_params["batch_size"]
num_classes = main_params["num_classes"]
learning_rate = main_params["learning_rate"]
num_epochs = main_params["num_epochs"]
dropout = main_params["dropout"]
weight_decay = main_params["weight_decay"]
early_stopping = main_params["early_stopping"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_transforms = transform()

def main():

    data = torchvision.datasets.ImageFolder(
        root="train", transform=all_transforms
    )

    p = main_params["train_size"]
    train_set, test_set = random_split(
        data,
        (int(p * len(data)), len(data) - int(p * len(data))),
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )

    model=MyVit((1,28,28),n_patches=7,n_blocks=2,hidden_d=8,n_heads=2,out_d=2).to(device)
    criterion = nn.CrossEntropyLoss()

    # Set optimizer with optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # We use the pre-defined number of epochs to
    # determine how many iterations to train the network on
    logging.warning(
        "Fitting of the model has begun"
    )
    for epoch in range(num_epochs):
        # Load in the data in batches using the train_loader object
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            correct_test = 0
            total_test = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                test_loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

            logging.info(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {np.round(loss.item(),3)}. Accuracy of the network on the test set: {np.round(100 * correct_test / total_test,3)} %"
            )

def patchify(images,n_patches):
    n,c,h,w=images.shape
    assert h==w, "Patchify method is implemented for square images only"

    patches=torch.zeros(n,n_patches**2,h*w//n_patches ** 2)
    patch_size=h//n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch=image[:, i*patch_size:(i+1)*patch_size, j*patch_size: (j+1)*patch_size]
                patches[idx, i*n_patches + j] = patch.flatten()
    return patches

def get_positional_embeddings(sequence_length, d):
    result=torch.ones(sequence_length,d)

    for i in range(sequence_length):
        for j in range(d):
            result[i][j]=np.sin(i/(10000**(j/d))) if j%2==0 else np.cos(i/(10000**(j-1)/d))
    return result

class MyMSA(nn.Module):
    def __init__(self,d,n,n_heads=2):
        super(MyMSA,self).__init__()
        self.d=d
        self.n_heads=n_heads

        assert d%n_heads==0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head=int(d/n_heads)

        self.q_mappings=nn.ModuleList([nn.Linear(d_head,d_head) for _ in range(self.n_heads)])
        self.k_mappings=nn.ModuleList([nn.Linear(d_head,d_head) for _ in range(self.n_heads)])
        self.v_mappings=nn.ModuleList([nn.Linear(d_head,d_head) for _ in range(self.n_heads)])

        self.d_head=d_head
        self.softmax=nn.Softmax(dim=-1)

    def forward(self,sequences):
        result=[]

        for sequence in sequences:
            seq_result=[]
            for head in range(self.n_heads):
                q_mapping=self.q_mappings[head]
                k_mapping=self.k_mappings[head]
                v_mapping=self.v_mappings[head]

                seq=sequence[:,head*self.d_head: (head+1)*self.d_head]
                q,k,v=q_mapping(seq),k_mapping(seq),v_mapping(seq)

                attention=self.softmax(q@k.T/(self.d_head**0.5))
                seq_result.append(attention@v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r,dim=0) for r in result])

class MyViTBlock(nn.Module):
    def __init__(self,hidden_d,n_heads,mlp_ratio=4):
        super(MyViTBlock,self).__init__()

        self.hidden_d=hidden_d
        self.n_heads=n_heads

        self.norm1=nn.LayerNorm(hidden_d)
        self.mhsa=MyMSA(hidden_d,n_heads)
        self.norm2=nn.LayerNorm(hidden_d)
        self.mlp=nn.Sequential(
            nn.Linear(hidden_d,mlp_ratio*hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio*hidden_d,hidden_d)
        )

    def forward(self,x):
        out=x+self.mhsa(self.norm1(x))
        out=out+self.mlp(self.norm2(out))
        return out

class MyVit(nn.Module):
    def __init__(self, chw=(1,28,28),n_patches=7, hidden_d=8,n_blocks=2,n_heads=2, out_d=2):
        super(MyVit,self).__init__()

        #Attributes

        self.chw = chw
        self.n_patches=n_patches
        self.hidden_d=hidden_d
        self.n_blocks=n_blocks
        self.n_heads=n_heads


        assert chw[1]%n_patches==0, "Input shape not entirely divisible by n_patches"
        assert chw[2]%n_patches==0, "Input shape not entirely divisible by n_patches"
        self.patch_size=(chw[1]/n_patches,chw[2]/n_patches)

        #1) Linear mapper

        self.input_d=int(chw[0]*self.patch_size[0]*self.patch_size[1])
        self.linear_mapper=nn.Linear(self.input_d,self.hidden_d)

        self.class_token=nn.Parameter(torch.rand(1,self.hidden_d))

        self.pos_embed=nn.Parameter(torch.tensor(get_positional_embeddings(self.n_patches**2 + 1, self.hidden_d)))
        self.pos_embed.requires_grad=False

        self.blocks=nn.ModuleList([MyViTBlock(hidden_d,n_heads) for _ in range(n_blocks)])

        self.mlp=nn.Sequential(
            nn.Linear(self.hidden_d,out_d),
            nn.Softmax(dim=-1)
        )
    def forward(self,images):
        n, c, h, w= images.shape
        patches=patchify(images,self.n_patches)
        tokens=self.linear_mapper(patches)

        #Adding classification token to the tokens

        tokens=torch.stack([torch.vstack((self.class_token,tokens[i])) for i in range(len(tokens))])

        pos_embed=self.pos_embed.repeat(n,1,1)
        out=tokens+pos_embed

        for block in self.blocks:
            out=block(out)

        out=out[:,0]

        return self.mlp(out)

if __name__=="__main__":
    main()
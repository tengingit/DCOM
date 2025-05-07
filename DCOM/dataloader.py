import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import scipy.io as scio
import os
from sklearn.model_selection import KFold


datasetdir = "../datasets"
class MDCDataset(Dataset):
    def __init__(self, datadir = datasetdir):
        datafile = os.path.join(datadir, self.name()+'.mat')
        self.data = scio.loadmat(datafile)
        self.features = torch.from_numpy(self.data['data'][0][0][1]).type(torch.float)
        self.labels = torch.from_numpy(self.data['target']).type(torch.long)-1    
        self.num_dim = self.labels.size(1)
        self.num_training = self.labels.size(0)
    
    def get_data(self):
        return self.features, self.labels

    def idx_cv(self, fold):
        '''
        fold: 0,1,...,9
        '''
        train_idx = self.data['idx_folds'][fold][0]['train'][0][0].reshape(-1).astype(np.int32)-1
        test_idx = self.data['idx_folds'][fold][0]['test'][0][0].reshape(-1).astype(np.int32)-1
        
        return train_idx, test_idx
    
    # def _power_set(self):
    #     return torch.unique(self.labels,dim=0,return_inverse=True)  #(Y_unique, labels_cp)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
    def __len__(self):
        return self.num_training
    
    @classmethod
    def name(cls):
        return cls.__name__
    
class MDCDataset1(MDCDataset):
    def __init__(self, datadir=datasetdir):
        datafile = os.path.join(datadir, self.name()+'.mat')
        self.data = scio.loadmat(datafile)
        self.feature = torch.from_numpy(self.data['data'][0][0][0]).type(torch.float)
        self.label = torch.from_numpy(self.data['target']).type(torch.long)-1    
        self.num_dim = self.label.size(1)
        self.num_training = self.label.size(0)

    
to_tensor=transforms.Compose([
        # transforms.Resize((256,256),2),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

to_tensor1=transforms.Compose([
        # transforms.Resize((256,256),2),
        transforms.Resize([224,224]),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

class ImageMDCDataset(Dataset):
    def __init__(self, datadir=datasetdir, train=True, transform=None):
        self.dataset_dir = os.path.join(datadir,self.name())
        image_dir = os.path.join(self.dataset_dir, "images")
        target_path = os.path.join(self.dataset_dir, "targets.csv")
        namelist_path = os.path.join(self.dataset_dir, "image_name.mat")
        namelist = scio.loadmat(namelist_path)['name']
        self.transform = transform
        self.image_paths = []
        with open(target_path,encoding = 'utf-8') as f:
            target = np.loadtxt(f,delimiter = ",")
        self.labels = torch.from_numpy(target).type(torch.long)   
        if train is not None:
            train_idx, test_idx = self.idx_cv(fold=0)
            if train:
                namelist = namelist[train_idx]
                self.labels = self.labels[train_idx]
            elif not train:
                namelist = namelist[test_idx]
                self.labels = self.labels[test_idx]
            for image in namelist:
                image_path = os.path.join(image_dir,image.strip())
                self.image_paths.append(image_path)
        for image in namelist:
            image_path = os.path.join(image_dir,image.strip())
            self.image_paths.append(image_path)
        self.num_dim = self.labels.size(1)
        self.num_data = self.labels.size(0)

    def get_data(self):
        # return self.features, self.labels
        return self.labels

    def idx_cv(self, fold):
        '''
        fold: 0,1,...,9
        '''
        train_idx_file = os.path.join(self.dataset_dir,"idx_cv_train.npy")
        test_idx_file = os.path.join(self.dataset_dir,"idx_cv_test.npy")
        if not os.path.exists(train_idx_file):
            self._cross_spilit()
        train_idx_dict = np.load(train_idx_file, allow_pickle=True).item()
        train_idx = train_idx_dict[fold]
        test_idx_dict = np.load(test_idx_file, allow_pickle=True).item()
        test_idx = test_idx_dict[fold]
            
        return train_idx, test_idx
    
    def _cross_spilit(self,nfold=10, shuffle=True):
        kf = KFold(n_splits=nfold, shuffle=shuffle)
        Y  = self.get_data()

        train_idx_dict = {}
        test_idx_dict = {}
        fold = 0
        for train_idx, test_idx in kf.split(Y):
            train_idx_dict[fold] = train_idx
            test_idx_dict[fold] = test_idx
            fold += 1

        np.save(self.dataset_dir+"\idx_cv_train.npy", train_idx_dict)
        np.save(self.dataset_dir+"\idx_cv_test.npy", test_idx_dict)
        # scio.savemat(self.dataset_dir+"\idx_cv_train.mat", )

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        img = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, self.labels[index]
    
    def __len__(self):
        return self.num_data
    
    @classmethod
    def name(cls):
        return cls.__name__  

class EncodedMDCDataset(ImageMDCDataset):
    def __init__(self, datadir=datasetdir, train=True, transform=None):
        self.dataset_dir = os.path.join(datadir,self.name())
        feature_path = os.path.join(self.dataset_dir, "data_Resnet18_pretrained.mat")
        target_path = os.path.join(self.dataset_dir, "targets.csv")
        data = scio.loadmat(feature_path)
        train_idx, test_idx = self.idx_cv(fold=0)
        with open(target_path,encoding = 'utf-8') as f:
            target = np.loadtxt(f,delimiter = ",")
        self.labels = torch.from_numpy(target).type(torch.long)   
        if train:
            self.features = torch.from_numpy(data['Train']).type(torch.float) 
            self.labels = self.labels[train_idx]
        else:
            self.features = torch.from_numpy(data['Test']).type(torch.float) 
            self.labels = self.labels[test_idx]
  
        self.num_dim = self.labels.size(1)
        self.num_data = self.labels.size(0)

    def get_data(self):
        return self.features, self.labels

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    

def data_loader(dataset,fold,batch_size,shuffle=False):
    train_idx, test_idx = dataset.idx_cv(fold)
    train_fold = data.dataset.Subset(dataset, train_idx)
    test_fold = data.dataset.Subset(dataset, test_idx)  
    
    train_iter = data.DataLoader(dataset=train_fold, batch_size=batch_size, shuffle=shuffle)
    test_iter = data.DataLoader(dataset=test_fold, batch_size=batch_size, shuffle=shuffle)

    return  train_iter,test_iter

def image_data_loader(dataset_name,batch_size,transform=None,shuffle=True):
    train_set = eval(dataset_name)(train=True,transform=transform)
    test_set = eval(dataset_name)(train=False,transform=transform)
    
    train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
    test_loader = data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_set,test_set,train_loader,test_loader


image_size = 224


class DeepFashion(EncodedMDCDataset):
    pass

class Flickr25k(EncodedMDCDataset):
    def __init__(self, datadir=datasetdir, train=True, transform=None):
        self.dataset_dir = os.path.join(datadir,self.name())
        feature_path = os.path.join(self.dataset_dir, "data_Resnet18_pretrained.mat")
        target_path = os.path.join(self.dataset_dir, "targets.csv")
        data = scio.loadmat(feature_path)
        idx_mat = os.path.join(self.dataset_dir, "idx_folds.mat")
        self.idx_folds = scio.loadmat(idx_mat)
        train_idx, test_idx = self.idx_cv(fold=0)
        with open(target_path,encoding = 'utf-8') as f:
            target = np.loadtxt(f,delimiter = ",")
        self.labels = torch.from_numpy(target).type(torch.long)-1
        if train:
            self.features = torch.from_numpy(data['Train']).type(torch.float) 
            self.labels = self.labels[train_idx]
        else:
            self.features = torch.from_numpy(data['Test']).type(torch.float) 
            self.labels = self.labels[test_idx]
        self.num_dim = self.labels.size(1)
        self.num_data = self.labels.size(0)
    def idx_cv(self, fold):
        train_idx = self.idx_folds['idx_folds'][fold][0]['train'][0][0].reshape(-1).astype(np.int32)-1
        test_idx = self.idx_folds['idx_folds'][fold][0]['test'][0][0].reshape(-1).astype(np.int32)-1
        
        return train_idx, test_idx

class BP4D(EncodedMDCDataset):
    pass

class SEWA(EncodedMDCDataset):
    pass

class Adult(MDCDataset):
    pass

class BeLaE(MDCDataset):
    pass

class CoIL2000(MDCDataset):
    pass

class Default(MDCDataset):
    pass

class Disfa(MDCDataset):
    pass

class Edm(MDCDataset):
    pass

class Enb(MDCDataset):
    pass

class Fera(MDCDataset):
    pass

class Flare1(MDCDataset):
    pass

class Flickr(MDCDataset):
    pass

class Jura(MDCDataset):
    pass

class Oes10(MDCDataset):
    pass

class Oes97(MDCDataset):
    pass

class Pain(MDCDataset):
    pass

class Rf1(MDCDataset):
    pass

class Scm1d(MDCDataset):
    pass

class Scm20d(MDCDataset):
    pass

class Song(MDCDataset):
    pass

class Thyroid(MDCDataset):
    def __init__(self, datadir = datasetdir):
        datafile = os.path.join(datadir, self.name()+'.mat')
        self.data = scio.loadmat(datafile)
        self.features = torch.from_numpy(self.data['data'][0][0][2]).type(torch.float)
        self.labels = torch.from_numpy(self.data['target']).type(torch.long)-1    
        self.num_dim = self.labels.size(1)
        self.num_training = self.labels.size(0)

class TIC2000(MDCDataset):
    pass

class Voice(MDCDataset):
    pass

class WaterQuality(MDCDataset):
    pass

class WQanimals(MDCDataset):
    pass

class WQplants(MDCDataset):
    pass


if __name__ == '__main__':
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    # if hasattr(torch.cuda, 'empty_cache'):
    # torch.cuda.empty_cache()
    
    # S = SEWA()
    # dataset = eval("BP4D")(train=True,transform=None)
    # train_idx, test_idx = dataset.idx_cv(fold=0)
    # train_fold = data.dataset.Subset(dataset, train_idx)
    # dataset = eval("BP4D")(train=False,transform=None)
    # test_fold = data.dataset.Subset(dataset, test_idx)  
    # print(dataset.image_paths)
    # print(train_fold)
    # print(test_fold.image_paths)

    F = Flare1()
    train_iter, test_iter = data_loader(dataset=F, fold=0, batch_size=4)
    for fea, lab in train_iter:
        print(fea.shape)
        print(lab.shape)
        print(fea)
        break
    # X,Y = S.get_data()
    # print(X.size())
    # print(X.element_size()*X.nelement())
    # S = SEWA()
    # S.name()

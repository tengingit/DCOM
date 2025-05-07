import torch
import torch.nn as nn
import torchvision
from mlp import MLP
from utils.utils import Init_random_seed
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class LatentVariableGenerator(nn.Module):
#     def __init__(self, dim_z):
#         super(LatentVariableGenerator, self).__init__()
#         self.z_dim = dim_z
#         # self.mu = nn.Parameter(torch.randn((batch_size,self.z_dim), requires_grad=True))
#         # self.logvar = nn.Parameter(torch.randn((batch_size,self.z_dim), requires_grad=True))

#     def forward(self,batch_size):
#         """
#         Given a standard gaussian distribution epsilon ~ N(0,1),
#         we can sample the random variable z as per z = mu + sigma * epsilon
#         """
#         mu = nn.Parameter(torch.randn(size=(batch_size,self.z_dim), requires_grad=True))
#         logvar = nn.Parameter(torch.randn(size=(batch_size,self.z_dim), requires_grad=True))

#         Z = self.reparameterize(mu, logvar)
        
#         return Z
    
    # def reparameterize(self, mu, logvar):
    #     """
    #     Given a standard gaussian distribution epsilon ~ N(0,1),
    #     we can sample the random variable z as per z = mu + sigma * epsilon
    #     """
    #     eps = torch.randn_like(mu)                                                     # (b,z_dim)
    #     sigma = torch.exp(logvar * 0.5)
        
    #     return mu + sigma * eps  

class VarMDC(nn.Module):
    def __init__(self, configs, br=True, cp=True):
        super(VarMDC, self).__init__()
        self.rand_seed = configs['rand_seed']
        self.num_dim = configs['num_dim']
        self.hs = configs['hidden_size']
        self.z_dim = configs['dim_z']
        self.num_per_dim = configs['num_per_dim']
        self.br = br
        self.cp = cp
        # self.LatentVariableGenerator = LatentVariableGenerator(self.z_dim)
        self.encoder = MLP(configs['num_feature'], self.z_dim, [self.hs],                    # self.hs, int(self.hs/2)
                           batchNorm=False,dropout=True,nonlinearity='relu',with_output_nonlinearity=True)
        self.decoder = MLP(self.z_dim, configs['num_feature'], [self.hs],                      # int(self.hs/2), self.hs
                           batchNorm=False,dropout=True,nonlinearity='relu',with_output_nonlinearity=False)
        if br:
            classifier_dict = []
            for i in range(self.num_dim):
                classifier_dict.append(MLP(self.z_dim+configs['num_feature'],self.num_per_dim[i],[self.hs]
                                        ,batchNorm=False,dropout=True,nonlinearity='relu',with_output_nonlinearity=False))
            self.classifiers = nn.ModuleList(classifier_dict)

        if cp:
            self.joint_classifier = MLP(self.z_dim+configs['num_feature'],configs['num_valid']+1,[self.hs],      # self.hs                          
                                batchNorm=False,dropout=True,nonlinearity='relu',with_output_nonlinearity=False)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        Init_random_seed(self.rand_seed)
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

        if self.br:
            for classifier in self.classifiers:
                classifier.reset_parameters() 
        if self.cp:
            self.joint_classifier.reset_parameters() 
              
    def forward(self, X):
        Z = self.encoder(X)
        # mu_logvar = self.decoder(Z).sigmoid_()
        # X_hat, logvar = mu_logvar.chunk(2,dim=1)
        X_hat = self.decoder(Z).sigmoid_()
        XZ = torch.cat((X,Z),dim=1)                        # (b, num_feature+dim_z)
        pred_probs = []
        if self.br:
            for i in range(self.num_dim):
                output = self.classifiers[i](XZ)
                pred_probs.append(output)

            if self.cp:
                joint_probs = self.joint_classifier(XZ)
                # return Z, X_hat, logvar, pred_probs, joint_probs
                return Z, X_hat, pred_probs, joint_probs
        else:
            if self.cp:
                joint_probs = self.joint_classifier(XZ)
                return X_hat, pred_probs, joint_probs

        return X_hat, pred_probs
    
    # def reparameterize(self, mu, logvar):
    #     """
    #     Given a standard gaussian distribution epsilon ~ N(0,1),
    #     we can sample the random variable z as per z = mu + sigma * epsilon
    #     """
    #     sigma = torch.exp(logvar * 0.5)
    #     eps = torch.randn_like(sigma)
        
    #     return mu + sigma * eps  
    
    
ResNet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
# ResNet = torchvision.models.resnet18(weights=None)
# ResNet.eval()
# ResNet.avgpool = nn.AdaptiveAvgPool2d((1, 1))
backbone = list(ResNet.children())[:-1]    #去掉全连接层
encode_Res = nn.Sequential(*backbone).to(device)

class MyImagenet(VarMDC):
    def __init__(self, configs, br=True, cp=True):
        super(MyImagenet, self).__init__(configs, br=br, cp=cp)
        self.backbone = encode_Res

    def forward(self, X):
        X = self.backbone(X).squeeze()
        Z = self.encoder(X)
        X_hat = self.decoder(Z).sigmoid_()
        XZ = torch.cat((X,Z),dim=1)                        # (b, num_feature+dim_z)
        pred_probs = []
        if self.br:
            for i in range(self.num_dim):
                output = self.classifiers[i](XZ)
                pred_probs.append(output)

        if self.cp:
            joint_probs = self.joint_classifier(XZ)
            return Z, X_hat, pred_probs, joint_probs

        return Z, X_hat, pred_probs

    def get_embs_backbone(self, X):
        embeddings_backbone = self.backbone(X).squeeze()
        return embeddings_backbone


# class BrImagenet(Brnet):
#     def __init__(self, configs, residual=True, optional=True):
#         super(BrImagenet, self).__init__(configs, residual=residual, optional=optional)
#         self.backbone = encode_Res

#     def forward(self, X):
#         X = self.backbone(X).squeeze()
#         embeddings_list = super(BrImagenet, self).forward(X)

#         return embeddings_list

#     def get_embs_backbone(self, X):
#         embeddings_backbone = self.backbone(X).squeeze()
#         return embeddings_backbone

# class CpImagenet(nn.Module):
#     def __init__(self, configs):
#         super(CpImagenet, self).__init__()
#         self.mlp = MLP(configs['num_feature'],configs['dim_emb'],[4*configs['dim_emb']],batchNorm=False,dropout=False,nonlinearity='relu',with_output_nonlinearity=False)
#         self.backbone = encode_Res

#     def forward(self, X):
#         X = self.backbone(X).squeeze()
#         embeddings = self.mlp(X)

#         return embeddings
    
#     def reset_parameters(self):
#         self.mlp.reset_parameters()
        

if __name__ == '__main__':
    # print(ResNet)
    # print("  ")
    # print(encode_Res)
    configs = {}
    configs['num_dim'] = 2
    configs['dim_z'] = 32
    configs['num_feature'] = 512
    configs['hidden_size'] = 256
    configs['num_per_dim'] = [2,3]
    configs['num_valid'] = 10
    # model = VarMDC(configs)
    # for name,para in model.named_parameters():
    #     # if "backbone" in name:
    #     #     para.requires_grad = False
    #     print(name)
    #     print(para)
    #     break
    # model = LatentVariableGenerator(dim_z=16)
    # # X = torch.ones((32,32))
    # for i in range(2):
    #     Z = model(4)
    #     print(Z)
    #     Z = model(2)
    #     print(Z)
    


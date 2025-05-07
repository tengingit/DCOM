import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CondIndepenLoss(nn.Module):
    def __init__(self):
        super(CondIndepenLoss, self).__init__()

    def forward(self, Z, X, X_hat, valid_cp, Y_valid, joint_probs, pred_probs):
        joint_probs = nn.Softmax(dim=1)(joint_probs[:,:-1])
        log_prods = self._prod_of_mar(valid_cp, pred_probs)             # (b,num_valid)
        fZ = self._fZ(Z)                                                # (b,)
        XZ = self._XZ(X, X_hat)
        num_valid = valid_cp.size(0)
        idx_valid = torch.argwhere(Y_valid < num_valid).squeeze()
        # print(joint_probs.size())
        # print(log_prods.size())
        # loss_ci = torch.sum(joint_probs * fZ * torch.log(joint_probs/prods))     # (1)
        # loss_ci = torch.sum(joint_probs * torch.log(joint_probs/prods))
        # loss_mat = joint_probs * (torch.log(joint_probs+1e-8) - log_prods)              # (b,num_valid)
        # print(XZ.size())
        # print(joint_probs.size())
        loss_mat = joint_probs * XZ * fZ * (torch.log(joint_probs+1e-8) - log_prods)              # (b,num_valid)
        # loss_mat = joint_probs * fZ * (torch.log(joint_probs+1e-8) - log_prods)              # (b,num_valid)
        # print(loss_mat[idx_valid,Y_valid[idx_valid]].size())
        loss_ci = torch.sum(loss_mat[idx_valid,Y_valid[idx_valid]])
        # loss_ci = torch.sum(loss_mat)
        loss_ci = torch.abs(loss_ci)

        # if torch.isnan(torch.sum((torch.log(joint_probs) - log_prods))):
        #     print("??????????????")
        #     print((torch.log(joint_probs) - log_prods))

        # print(log_prods.size())
        # print(log_prods)
        # print(fZ.size())
        # print('')
        # print(fZ)
        # print(idx_valid)
        if idx_valid.dim() == 0:
            return loss_ci    
        elif len(idx_valid) == 0:
            return loss_ci
        #     print('No valid class for loss calculation')
        return loss_ci/len(idx_valid)
        # return loss_ci/joint_probs.size(0)
    
    def _prod_of_mar(self, valid_cp, pred_probs):
        log_prods = []
        for y in valid_cp:
            log_prod = 0
            for dim, pred_prob in enumerate(pred_probs):
                pred_prob = nn.Softmax(dim=1)(pred_prob)
                # pred_prob[pred_prob<1e-6] = 1e-6
                log_prod += torch.log(pred_prob[:, y[dim]]+1e-8)                   # (b,)
            # prod[prod<1e-8] = 1e-8                             # To avoid the product being 0
            log_prods.append(log_prod)                            

        return torch.stack(log_prods,dim=1)                        # (b,num_valid)    
    
    def _fZ(self, Z):
        # num_batch = Z.size(0)
        # sigma = torch.exp(logvar * 0.5)                       # (b,z_dim)
        # fZ1 = []
        # Z_nor = Z - mu
        # for i in range(num_batch):
        #     numerator = torch.exp(-0.5 * (torch.sum(Z_nor[i]*Z_nor[i]/(sigma[i]*sigma[i]))))
        #     # denominator = pow(2*torch.pi,Z_dim/2)*torch.exp(torch.sum(logvar[i]) * 0.5)
        #     denominator = torch.exp(torch.sum(logvar[i]) * 0.5)
        #     fZ1.append(numerator/denominator)
        # fZ1 = torch.stack(fZ1)
        # print(fZ1)
        # fZ = torch.exp(-0.5 * (torch.sum((Z-mu)*(Z-mu)/(sigma*sigma), dim=1) + torch.sum(logvar, dim=1)))
        # fZ = torch.exp(-0.5 * torch.sum(Z*Z, dim=1))/pow(2*torch.pi,Z.size(1)/2)
        fZ = torch.exp(-0.5 * torch.sum(Z*Z, dim=1))
        # print(fZ)

        return fZ.view(-1,1)                                     # (b,1)

    def _XZ(self, X, X_hat):
        # num_feature = X_hat.size(1)
        # XZ = num_feature * 0.798                                 # log(2pi)=0.798
        # XZ += torch.sum(logvar,dim=1)
        # XZ += torch.sum((X - X_hat) * (X - X_hat)/ torch.exp(logvar+1e-8),dim=1)
        # return (-0.5 * XZ / num_feature).view(-1,1)  
        XZ = torch.exp(-0.5 * torch.sum((X - X_hat) * (X - X_hat),dim=1))
        return XZ.view(-1,1)     



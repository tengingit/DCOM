import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_list=[], batchNorm=False, dropout=False,
                 nonlinearity='relu', negative_slope=0.1,
                 with_output_nonlinearity=False):
        super(MLP, self).__init__()
        self.nonlinearity = nonlinearity
        self.negative_slope = negative_slope
        self.fcs = nn.ModuleList()
        if hidden_list:
            in_dims = [input_size] + hidden_list
            out_dims = hidden_list + [output_size]
            for i in range(len(in_dims)):
                self.fcs.append(nn.Linear(in_dims[i], out_dims[i]))
                if with_output_nonlinearity or i < len(hidden_list):
                    if batchNorm:
                        self.fcs.append(nn.BatchNorm1d(out_dims[i], track_running_stats=True))
                    if nonlinearity == 'relu':
                        self.fcs.append(nn.ReLU(inplace=False))
                    elif nonlinearity == 'leaky_relu':
                        self.fcs.append(nn.LeakyReLU(negative_slope, inplace=True))  # Controls the angle of the negative slope (which is used for negative input values). Default: 1e-2
                    else:
                        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
                if dropout:
                    # if i == 0:
                        # self.fcs.append(nn.Dropout(p=0.2))
                    # elif i < len(hidden_list):
                    if i < len(hidden_list):
                        self.fcs.append(nn.Dropout(p=0.5))
        else:
            self.fcs.append(nn.Linear(input_size, output_size))
            if with_output_nonlinearity:
                if batchNorm:
                    self.fcs.append(nn.BatchNorm1d(output_size, track_running_stats=True))
                if nonlinearity == 'relu':
                    self.fcs.append(nn.ReLU(inplace=False))
                elif nonlinearity == 'leaky_relu':
                    self.fcs.append(nn.LeakyReLU(negative_slope, inplace=True))
                else:
                    raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
                if dropout:
                    self.fcs.append(nn.Dropout(p=0.5))

        self.reset_parameters()
        
    def reset_parameters(self):
        for l in self.fcs:
            # if isinstance(l, nn.Linear):
            #     nn.init.kaiming_uniform_(l.weight, a=self.negative_slope,
            #                              nonlinearity=self.nonlinearity)
            if isinstance(l, nn.Linear):
                nn.init.xavier_uniform_(l.weight)
            #     # if self.nonlinearity == 'leaky_relu' or self.nonlinearity == 'relu':
            #     #     nn.init.uniform_(l.bias, 0, 0.1)
            #     # else:
            #     #     nn.init.constant_(l.bias, 0.0)
            elif l.__class__.__name__ == 'BatchNorm1d':
                l.reset_parameters()
            # l.reset_parameters()
    def forward(self, input):
        for l in self.fcs:
            input = l(input)
        return input
    
class MLP_br(nn.Module):
    def __init__(self, num_dim, input_size, output_size, hidden_list=[], batchNorm=False,
                 dropout=False, nonlinearity='relu',with_output_nonlinearity=False):
        super(MLP_br, self).__init__()
        self.num_dim = num_dim
        self.mlps = nn.ModuleList()
        for _ in range(num_dim):
            self.mlps.append(MLP(input_size,output_size,hidden_list,batchNorm=batchNorm,
                                 dropout=dropout,nonlinearity=nonlinearity,
                                 with_output_nonlinearity=with_output_nonlinearity))

        self.reset_parameters()

    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()

    def forward(self, X):
        embeddings = []
        for dim in range(self.num_dim):
            embeddings.append(self.mlps[dim](X))       #(batch_size,dim_emb)
    
        return torch.cat(embeddings,dim=1)             #(batch_size,q*dim_emb)
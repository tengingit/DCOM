# import sys
import os
import argparse
import torch
import pandas as pd
import time
import dataloader
from dataloader import *
from model import *
from utils.utils import Init_random_seed
from utils.config import *
from utils.metrics import eva
from loss_ci import CondIndepenLoss

parser = argparse.ArgumentParser()
parser.add_argument('-dataset','--dataset', type=str, default="DeepFashion", help='dataset on which the experiment is conducted')
parser.add_argument('-bs', '--batch_size', type=int, default=512, help='batch size for one iteration during training')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2, help='learning rate parameter')
parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4, help='weight decay parameter')
parser.add_argument('-max_epoch', '--max_epoch', type=int, default=500, help='maximal training epochs')
parser.add_argument('-lambda', '--lambda_tradeoff', type=float, default=1, help='trade-off parameter for cross-entropy loss')
parser.add_argument('-beta', '--beta_tradeoff', type=float, default=1, help='trade-off parameter for conditional independence loss')
parser.add_argument('-hs', '--hidden_size', type=int, default=256, help='the dimensionality of hidden embeddings.')
parser.add_argument('-dz', '--dim_z', type=int, default=1024, help='the dimensionality of latent variable Z.')
parser.add_argument('-optional', '--optional', action='store_true', help='whether to use conditional independence loss')
parser.add_argument('-cuda', '--cuda', action='store_true', help='whether to use gpu')
parser.add_argument('-dft_cfg', '--default_cfg', action='store_true', help='whether to run experiment with default hyperparameters')
parser.add_argument('-test', '--test_mode', action='store_true', help='whether to use existing model for testing only')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args,nfold):
    Init_random_seed(seed=0)
    dataset_name = args.dataset
    print(dataset_name)
    # dataset = eval(dataset_name)(train=None,transform=None)      #dataset = BeLaE()
    # Y = dataset.get_data()
    # Y = Y.to(device)

    configs = generate_default_config()
    # configs['dataset'] = dataset          
    configs['weight_decay'] = args.weight_decay
    configs['lr'] = args.learning_rate
    configs['lambda_tradeoff'] = args.lambda_tradeoff
    configs['beta_tradeoff'] = args.beta_tradeoff
    configs['hidden_size'] = args.hidden_size
    configs['dim_z'] = args.dim_z
    # Loading dataset-specific configs
    if args.default_cfg:
        eval('{}_configs'.format(dataset_name))(configs)
    print(configs)
    criterion_ae = torch.nn.MSELoss()
    criterion_cls = torch.nn.CrossEntropyLoss()
    criterion_ci = CondIndepenLoss()

    results = np.zeros((3,10))
    for fold in range(0,nfold):
        # train_idx, test_idx = dataset.idx_cv(fold)
        train_set,test_set,train_loader,test_loader = dataloader.image_data_loader(dataset_name, batch_size=args.batch_size, shuffle=True)
        _, Y_train = train_set.get_data()
        X_test, Y_test = test_set.get_data()
        Y_train, Y_test = Y_train.to(device), Y_test.to(device)

        num_dim = Y_train.size(1)                 #number of dimensions(class variables)
        label_per_dim = {}                  #class labels in each dimension
        num_per_dim = np.zeros((num_dim),dtype = int)  #number of class labels in each dimension
        for dim in range(num_dim):
            labelset = torch.unique(Y_train[:,dim])
            label_per_dim[dim] = list(labelset)
            num_per_dim[dim] = len(label_per_dim[dim])
        configs['num_feature'] = X_test.size(1)   
        configs['num_dim'] = num_dim
        configs['label_per_dim'] = label_per_dim
        configs['num_per_dim'] = num_per_dim

        num_example = Y_train.size(0)
        Y_unique, _, counts = torch.unique(Y_train,dim=0,return_inverse=True, return_counts=True)
        if num_example < 1000:
            num_tail = 1
        elif num_example <10000:
            num_tail = 10
        else:
            num_tail = 100
        valid_cp = Y_unique[counts>num_tail]
        configs['num_valid'] = valid_cp.size(0)
        configs['valid_cp'] = valid_cp
        print(f"num_valid:{configs['num_valid']}")
        isValid = torch.nonzero((Y_train.unsqueeze(1) == valid_cp).all(dim=-1))          # (<m,num_valid)
        Y_valid_train = torch.zeros(Y_train.size(0),dtype=torch.long) + configs['num_valid']   # (m,)
        Y_valid_train = Y_valid_train.to(device)
        Y_valid_train[isValid[:,0]] = isValid[:,1]                                       # (m,)
        isValid = torch.nonzero((Y_test.unsqueeze(1) == valid_cp).all(dim=-1))          # (<m,num_valid)
        Y_valid_test = torch.zeros(Y_test.size(0),dtype=torch.long) + configs['num_valid']   # (m,)
        Y_valid_test = Y_valid_train.to(device)
        Y_valid_test[isValid[:,0]] = isValid[:,1] 
       
        criterion_cls_cp = torch.nn.CrossEntropyLoss(ignore_index=configs['num_valid'])
        criterion_list = [criterion_ae,criterion_cls,criterion_cls_cp,criterion_ci]
        model = VarMDC(configs, br=True, cp=True)  
        model = model.to(device)         
        model.reset_parameters()
        # print(model)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=configs['weight_decay'],eps=1e-8)
        optimizer = torch.optim.SGD(model.parameters(), lr=configs['lr'], momentum=0.9, weight_decay=configs['weight_decay'])

        log_loss_path = "logs_ci/"+args.dataset+"/loss/fold"+str(fold)
        log_metric_path = "logs_ci/"+args.dataset+"/metric/fold"+str(fold)
        checkpoint_path = "checkpoints_ci/"+args.dataset+"/fold"+str(fold)
        result_loss_path = "results_ci/"+args.dataset+"/loss/fold"+str(fold)
        result_metric_path = "results_ci/"+args.dataset+"/metric/fold"+str(fold)
        result_path = "results_ci/"+args.dataset
        path_list = [log_loss_path, log_metric_path, checkpoint_path, result_loss_path, result_metric_path]
        for path in path_list:
            if not os.path.exists(path):
                os.makedirs(path)
        file_saver = "/{ds}_hs{hs}_dz{dz}_lam{lam}_be{beta}_lr{lr}_wd{wd}".format(ds=args.dataset,
                                                                                    hs=args.hidden_size,
                                                                                    dz=args.dim_z,
                                                                                    lam = args.lambda_tradeoff,
                                                                                    beta = args.beta_tradeoff,
                                                                                    lr=args.learning_rate,
                                                                                    wd=args.weight_decay)
        log_loss_table = np.zeros(shape=(args.max_epoch, 4))
        log_metric_table = np.zeros(shape=(args.max_epoch, 3))
        results_loss_table = np.zeros(shape=(args.max_epoch, 4))
        results_metric_table = np.zeros(shape=(args.max_epoch, 3))

        # train_loader, test_loader = data_loader(dataset, fold, batch_size=args.batch_size, shuffle=False)
        model_path = checkpoint_path + file_saver +".pth"
        if args.test_mode and os.path.exists(model_path):
            print('Loading existing models O.o')
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['state_dict'])
            test_loss, loss_ae_item, loss_cls_item, loss_ci_item, pred_Y_test = predict(test_loader, model, configs, Y_valid_test, criterion_list, epoch=args.max_epoch)
            test_ham, test_exa, test_sub  = eva(Y_test, pred_Y_test)
            test_ham, test_exa, test_sub = test_ham.cpu(), test_exa.cpu(), test_sub.cpu()
        else:
            print('Fold'+str(fold)+': start training!')
            best_ham, best_epoch = 0, 0
            for epoch in range(args.max_epoch):
                train_loss, loss_ae_item, loss_cls_item, loss_ci_item, pred_Y_train = train(train_loader, model, optimizer, configs, Y_valid_train, criterion_list, epoch)
                train_ham, train_exa, train_sub = eva(Y_train, pred_Y_train)
                log_loss_table[epoch, :] = train_loss, loss_ae_item, loss_cls_item, loss_ci_item
                log_metric_table[epoch, :] = train_ham.cpu(), train_exa.cpu(), train_sub.cpu()
                if (epoch+1) % 2 == 0:
                    print('[{}/{}] Training with CI Loss:'.format(epoch + 1, args.max_epoch))
                    print(f"train_loss:{train_loss}, loss_ae:{loss_ae_item}, loss_cls:{loss_cls_item}, loss_ci:{loss_ci_item},\ntrain_ham:{train_ham}, train_exa:{train_exa}, train_sub:{train_sub}")
                np.savetxt(log_loss_path+file_saver+".csv", log_loss_table, delimiter=',', fmt='%1.4f')
                np.savetxt(log_metric_path+file_saver+".csv", log_metric_table, delimiter=',', fmt='%1.4f')
            
                test_loss, loss_ae_item, loss_cls_item, loss_ci_item, pred_Y_test = predict(test_loader, model, configs, Y_valid_test, criterion_list, epoch)
                test_ham, test_exa, test_sub  = eva(Y_test, pred_Y_test)
                test_ham, test_exa, test_sub = test_ham.cpu(), test_exa.cpu(), test_sub.cpu()
                results_loss_table[epoch, :] = test_loss, loss_ae_item, loss_cls_item, loss_ci_item
                results_metric_table[epoch, :] = test_ham, test_exa, test_sub
                np.savetxt(result_loss_path+file_saver+".csv", results_loss_table, delimiter=',', fmt='%1.4f')
                np.savetxt(result_metric_path+file_saver+".csv", results_metric_table, delimiter=',', fmt='%1.4f')

                if test_ham > best_ham:
                    best_ham = test_ham
                    best_epoch = epoch
            # save model of the last epoch
            torch.save({'best_epoch': best_epoch+1, 'best_ham': best_ham, 'state_dict': model.state_dict()}, model_path)
        results[:,fold] = test_ham, test_exa, test_sub
                
    df = pd.DataFrame(results,index=['hammingscore','exactmatch','subexactmatch'])
    df = df.T
    df.to_csv(result_path+file_saver+".csv")


def train(train_loader, model, optimizer, configs, Y_valid_train, criterion_list, epoch):
    model.train()
    train_loss, loss_ae_item, loss_cls_item, loss_ci_item = 0, 0, 0, 0
    pred_Y = []
    for i, (X, Y) in enumerate(train_loader):
        pred_Y_batch = []
        X, Y = X.to(device), Y.to(device)
        num_dim = Y.size(1)
        batchsize = Y.size(0)
        X_n = add_noise(X)
        Z, X_hat, pred_probs, joint_probs = model(X_n)
        loss_ae = criterion_list[0](X_hat,X)
        # loss_ae /= batchsize
        loss_cls = torch.tensor(0, dtype=torch.float32).to(device)
        for dim, prob in enumerate(pred_probs):
            loss_cls += criterion_list[1](prob,Y[:,dim])
            pred_Y_batch.append(torch.argmax(prob, dim=1, keepdim=True))                  # (n,1)
        loss_cls /= num_dim
        Y_valid = Y_valid_train[i*batchsize:(i+1)*batchsize]
        loss_cls += criterion_list[2](joint_probs,Y_valid)

        loss_ci = torch.tensor(0, dtype=torch.float32).to(device)
        if epoch >= 0:
            loss_ci = criterion_list[3](Z, X, X_hat, configs['valid_cp'], Y_valid, joint_probs, pred_probs)
        pred_Y_batch = torch.cat(pred_Y_batch,dim=1)                                      #(n,q)
        pred_Y.append(pred_Y_batch)
        loss = loss_ae + configs['lambda_tradeoff'] * loss_cls + configs['beta_tradeoff'] * loss_ci
        optimizer.zero_grad()    
        loss.backward()          
        optimizer.step()         
        train_loss += loss.item()
        loss_ae_item += loss_ae.item()
        loss_cls_item += loss_cls.item()
        loss_ci_item += loss_ci.item()

    pred_Y = torch.cat(pred_Y, dim=0)

    return train_loss, loss_ae_item, loss_cls_item, loss_ci_item, pred_Y

def predict(test_loader, model, configs, Y_valid_test, criterion_list, epoch):
    pred_Y = []
    test_loss, loss_ae_item, loss_cls_item, loss_ci_item = 0, 0, 0, 0
    with torch.no_grad():
        model.eval()
        for i, (X, Y) in enumerate(test_loader):
            pred_Y_batch = []
            X, Y = X.to(device), Y.to(device)
            batchsize = Y.size(0)
            num_dim = Y.size(1)
            X_n = add_noise(X)
            Z, X_hat, pred_probs, joint_probs = model(X_n)
            loss_ae = criterion_list[0](X_hat,X) 
            # loss_ae /= batchsize
            loss_cls = torch.tensor(0, dtype=torch.float32).to(device)
            for dim, prob in enumerate(pred_probs):
                loss_cls += criterion_list[1](prob,Y[:,dim])
                pred_Y_batch.append(torch.argmax(prob, dim=1, keepdim=True))                  #(n,1)
            loss_cls /= num_dim
            Y_valid = Y_valid_test[i*batchsize:(i+1)*batchsize]
            loss_cls += criterion_list[2](joint_probs,Y_valid)
            loss_ci = torch.tensor(0, dtype=torch.float32).to(device)
            if epoch >= 0:
                loss_ci = criterion_list[3](Z, X, X_hat, configs['valid_cp'], Y_valid, joint_probs, pred_probs)
            pred_Y_batch = torch.cat(pred_Y_batch,dim=1)                                      #(n,q)
            pred_Y.append(pred_Y_batch)

            loss = loss_ae + configs['lambda_tradeoff'] * loss_cls + configs['beta_tradeoff'] * loss_ci      
            test_loss += loss.item()
            loss_ae_item += loss_ae.item()
            loss_cls_item += loss_cls.item()
            loss_ci_item += loss_ci.item()

    pred_Y = torch.cat(pred_Y, dim=0)

    return test_loss, loss_ae_item, loss_cls_item, loss_ci_item, pred_Y

def add_noise(inputs,noise_factor=1e-5):
    noisy = inputs+torch.randn_like(inputs) * noise_factor
    noisy = torch.clip(noisy,0.,1.)
    return noisy

# def add_noise(data, frac):
#     """
#     data: Tensor
#     frac: fraction of unit to be masked out
#     """
#     data_noise = data.clone()
#     rand = torch.rand(data.size())
#     data_noise[rand<frac] = 0
#     return data_noise

if __name__ == '__main__':
    args = parser.parse_args()
    start_time = time.time()

    main(args,nfold=1)
 
    end_time = time.time()    
    print("during {:.2f}s".format(end_time - start_time))



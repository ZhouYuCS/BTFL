# -*- coding: utf-8 -*-
import torch
import copy
import itertools
import time
from pcode.utils.tensor_buffer import TensorBuffer
from scipy.stats import beta
from scipy.special import iv
from scipy.stats import multivariate_normal
import pcode.utils.loss as loss
import torch.distributed as dist
from pcode.local_training.base_worker import BaseWorker
import pcode.create_dataset as create_dataset
from pcode.utils.stat_tracker import RuntimeTracker
import pcode.local_training.utils as utils
import pcode.create_optimizer as create_optimizer
import torch.nn.functional as F
import torch.nn as nn
from pcode.datasets.aug_data import aug
import torchvision.transforms as transforms
import math
import numpy as np
def get_label_number(name):
        if name=="cifar10":
            return 10
        elif name=="oh":
            return 65
        elif name=="imagenet" :
            return 1000
        else:
            return 100
        
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

class CategoryDistributionModel(nn.Module):
    def __init__(self, input_dim,q_level):
        super(CategoryDistributionModel, self).__init__()
        self.category_parameters = nn.Parameter(torch.rand(q_level,input_dim))  # 用随机初始化的参数表示类别分布
        self.q_level=q_level
        self.input_dim=input_dim
    def forward(self, x,log=True):
        #many_out=self.category_parameters.repeat(x.shape[0],1,1)  
        test_outputs = torch.gather(self.category_parameters, 0, x)
        if log:
             test_outputs=torch.log(test_outputs*(1-0.2*self.q_level)+0.2)
        score=test_outputs
        return_value=score.sum(1)
        if not log:
            return_value=score.mean(1)
        return return_value.unsqueeze(1).detach()
    
class BTFLWorker(BaseWorker):
    def __init__(self, conf, is_fine_tune=False):
        super(BTFLWorker, self).__init__(conf)
        self.conf = conf
        self.n_personalized_epochs = self.conf.n_personalized_epochs
        self.eval_dataset = self.conf.eval_dataset  # can be test_loader or val_loader
        self.is_in_childworker = True
        self.update_condition=self.conf.update_condition
        # test-time self-supervised aggregation
        self.num_head = 2
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.temperature=self.conf.temperature
        self.THE_steps = self.conf.THE_steps
        self.agg_weight = torch.nn.Parameter(torch.rand((self.conf.batch_size, self.num_head)).cuda(), requires_grad=False)
        self.agg_weight.data.fill_(1 / self.num_head)
        self.alpha = self.conf.THE_alpha
        self.beta = self.conf.THE_beta
        self.kb=self.conf.kb
        self.mu=self.conf.mu
        self.nu=self.conf.nu
        self.sigma=conf.sigma
       
    
        self.is_tune_net = is_fine_tune
        self.is_rep_history_reused = self.conf.is_rep_history_reused
        self.q_level=conf.q_level
        self.pro= CategoryDistributionModel(self.conf.rep_len,self.q_level).to(self.graph.device)
       
        self.label_number=get_label_number(self.conf.data)

    
    
    def run(self):
        while True:
            self._listen_to_master()

            # check if we need to terminate the training or not.
            if self._terminate_by_early_stopping():
                return

            # receive the perform standard local training.
            self._recv_model_from_master()

            # create personal model and register hook.
            if not hasattr(self, "personal_head"):
                self.personal_head = nn.Linear(self.conf.rep_len, utils.get_num_classes(self.conf.data), bias=False)

            self._pro_train(model=self.model)
            self.send_pro()
            self._recv_pro_from_master()
            #self._recv_pros_from_master()
            # personalization.
            p_state = self._personalized_train(model=self.model)
            params_to_send = copy.deepcopy(p_state["model"].state_dict())  # use deepcopy

            #new
            perf = [0.0] * 6
            perf.extend(self.local_rep.cpu().squeeze().tolist())
            perf.extend(self.local_rep_sigma.cpu().squeeze().tolist())
            self._send_model_to_master(params_to_send, perf)
            self.global_rep = torch.ones((self.conf.rep_len*self.conf.n_clients,))
            dist.broadcast(tensor=self.global_rep, src=0)
            dist.barrier()

            self.global_rep = self.global_rep.cuda()
            self.global_rep_sigma = torch.ones((self.conf.rep_len*self.conf.n_clients,)).cuda()
            
            
            


            self.get_rep()
            state = self._brm_train(model=self.model)
            params_to_send = copy.deepcopy(state["model"].state_dict())
            perf = [0.0] * 6
           
               
            perf = self._evaluate_all_test_sets(p_state)
              
            # display the personalized perf and sync the model & perf.
            self._display_info(p_state, perf)

            # also send local rep and sigma
            perf.extend(self.local_rep.cpu().squeeze().tolist())
            perf.extend(self.local_rep_sigma.cpu().squeeze().tolist())
            self._send_model_to_master(params_to_send, perf)
            del state, p_state

            if self._terminate_by_complete_training():
                return

    def _brm_train(self, model):
      
        state = self.train(model)
       
        return state

    def _personalized_train(self, model):
        self.is_in_personalized_training = True
        store = self.n_local_epochs
        self.n_local_epochs = self.n_personalized_epochs
        self.erm_criterion = nn.CrossEntropyLoss(reduction="mean")
        # define dataloader, optimizer, scheduler and tracker
        state = self._init_train_process(model=model)
        self.personal_head.to(self.graph.device)
        #self.personal_head.weight.data=state["model"].classifier.weight.data/2+self.personal_head.weight.data/2
    
        # we want to optimize personal head
        state["optimizer"] = create_optimizer.define_optimizer(
            self.conf, model=self.personal_head, optimizer_name=self.conf.optimizer, lr=self._get_round_lr()
        )
        # freeze the model, except the personal head
        state["model"].requires_grad_(False)
        self.personal_head.requires_grad_(True)
        self.rep_layer = utils.determine_hook(state["model"], name=self.conf.arch)
        self.rep_layer.register_forward_hook(utils.hook)
        while not self._is_finished_one_comm_round(state):
            self.local_rep = []      
            self.per_class_rep = {i: [] for i in range(utils.get_num_classes(self.conf.data))}
            self.per_class_number={i:0 for i in range(utils.get_num_classes(self.conf.data))}
            for _input, _target in state["train_loader"]:
                # load data
                with self.timer("load_data", epoch=state["scheduler"].epoch_):
                    data_batch = create_dataset.load_data_batch(
                        self.conf, _input, _target, is_training=True,
                    )

                # inference and get current performance.
                with self.timer("forward_pass", epoch=state["scheduler"].epoch_):
                    state["optimizer"].zero_grad()
                    g_out = state["model"](data_batch["input"])
                    p_out = self.personal_head(self.rep_layer.rep)
                    teacher_prob = F.softmax(p_out, dim=1)
                    student_log_prob = F.log_softmax(g_out, dim=1)
                    loss = self.erm_criterion(p_out, data_batch["target"])#+self.sigma*((-g_out.softmax(1) * p_out.log_softmax(1)).sum(1)).mean()
                    agg_out = torch.stack([g_out, p_out], dim=1).mean(dim=1)
                    performance = self.metrics.evaluate(loss, agg_out, data_batch["target"])
                    state["tracker"].update_metrics(
                        [loss.item()] + performance, n_samples=data_batch["input"].size(0)
                    )
                    for i, label in enumerate(data_batch["target"]):
                        self.per_class_rep[label.item()].append(self.rep_layer.rep[i, :].unsqueeze(0))
                        self.per_class_number[label.item()]+=1
                with self.timer("backward_pass", epoch=state["scheduler"].epoch_):
                    loss.backward()
                    state["optimizer"].step()
                    state["scheduler"].step()

                if self.conf.display_log:
                    self._display_logging(state)
                if self._is_diverge(state):
                    self._terminate_comm_round(state)
                    return state

            # refresh the logging cache at the end of each epoch.
            state["tracker"].reset()
            if self.logger.meet_cache_limit():
                self.logger.save_json()

        # terminate
        self._compute_prototype()
        self.local_rep_sigma=torch.ones_like(self.local_rep)
        self._compute_prototype_base_entropy(state)
        self._terminate_comm_round(state)
        self.n_local_epochs = store
        self.is_in_personalized_training = False
        return state

    def _compute_prototype(self):
        # compute the average representation of local training set.
        for (k, v) in self.per_class_rep.items():
            if len(v) != 0:
                self.local_rep.append(torch.cat(v).cuda())
        self.local_rep = torch.cat(self.local_rep).mean(dim=0).cuda()
        
    def _compute_prototype_sigma(self,state):
        for _input, _target in state["train_loader"]:
            data_batch = create_dataset.load_data_batch(
                        self.conf, _input, _target, is_training=True,
                    )
            _ = state["model"](data_batch["input"])
            self.local_rep_sigma.append(torch.pow(self.rep_layer.rep-self.local_rep,2))
        self.local_rep_sigma=torch.cat(self.local_rep_sigma).mean(dim=0).cuda()
        self.local_rep_sigma=torch.ones_like(self.local_rep_sigma)
       
    def _compute_prototype_base_entropy(self,state): 
        self.local_en=[]
        self.global_en=[]
        for _input, _target in state["train_loader"]:
            data_batch = create_dataset.load_data_batch(
                        self.conf, _input, _target, is_training=True,
                    )
            g_out = state["model"](data_batch["input"])
            p_out = self.personal_head(self.rep_layer.rep)
            self.local_en.append(-(p_out.softmax(1) * p_out.log_softmax(1)).sum(1))
            self.global_en.append(-(g_out.softmax(1) * g_out.log_softmax(1)).sum(1))
        self.local_en=torch.cat(self.local_en).mean(dim=0).item()
        self.global_en=torch.cat(self.global_en).mean(dim=0).item()

    def within_integral(self,x,lam):
        fz=self.beta_distribution.pdf(x)*x
        fm=x+(1-x)*lam
        return fz/fm
    
    def _calculate_samplewise_weight(self,rep,g_out,p_out):
        
        self.g_entropy=-(g_out.softmax(1) * g_out.log_softmax(1)).sum(1)
        self.l_entropy=-(p_out.softmax(1) * p_out.log_softmax(1)).sum(1)
        self.g_Bol=torch.exp(-(self.g_entropy)/self.kb).cpu().numpy()
        self.l_Bol=torch.exp(-(self.l_entropy)/self.kb).cpu().numpy()
        self.div_bol=self.l_Bol/self.g_Bol
        dx=torch.ones(self.rep_layer.rep.shape[0]).numpy()
        self.div_bol = self.div_bol*dx
        self.div_entropy=(self.g_entropy/self.l_entropy).cpu().numpy()*dx
        self.abother_deal=(self.l_entropy-self.g_entropy).cpu().numpy()*dx
        '''
        local_kappa=torch.norm(self.local_rep).cpu().item()
        global_kappa=torch.norm(torch.from_numpy(self.sth)).item()
        vmf_local_fm=iv(31,local_kappa)
        vmf_global_fm=iv(31,global_kappa)
        '''
        sim_local=self.cos(self.rep_layer.rep,self.local_rep.unsqueeze(0)).cpu()
        #sim_global=self.cos(self.rep_layer.rep,torch.from_numpy(self.sth).unsqueeze(0).cuda()).cpu()
        self.vmf_local_fz= torch.exp(sim_local/self.temperature).numpy()
        self.vmf_global_fz=torch.zeros(self.rep_layer.rep.shape[0]).numpy()
       
        features_temp=self.fsq_quantize(rep,self.q_level).to(torch.int64)
            
            
            
        self.disc_local=self.pro(features_temp).cpu().numpy().flatten()
        self.disc_global=self.global_pro(features_temp).cpu().numpy().flatten()

        self.div_vmf= (self.disc_local/self.disc_global)
        self.div_vmf=torch.exp(self.pro(features_temp)-self.global_pro(features_temp)).cpu().numpy().flatten()
           
        

        delta_sl= -(self.local_en-self.l_entropy).cpu().numpy()/self.local_en
        delta_sg=-(math.log(100)-self.g_entropy).cpu().numpy()/math.log(100)

        
        self.pow_exp_local=np.exp(delta_sl)
        self.pow_exp_global=np.exp(delta_sg)
        

        self.dim_vmf=np.exp((self.disc_local*self.pow_exp_local-self.disc_global*self.pow_exp_global)/ self.conf.rep_len)
        
    
        self.lams_np=(self.kb*np.tanh(self.dim_vmf-1)+1)*(self.dim_vmf>1)+ (self.dim_vmf<1)*(1/(self.kb*np.tanh(1/self.dim_vmf-1)+1))
        #self.lams_np=self.dim_vmf
        #self.lams_np=self.div_vmf/self.div_vmf#=np.minimum(np.maximum((delta_sl>0)*0+(delta_sl<0)*self.dim_vmf,0.001),1000)
        # 积分
        a = 0.01
        b = 0.99
        n = 100
        integrals = np.zeros_like(self.lams_np)
        x = np.linspace(a, b, n)
        for i, lam in enumerate(self.lams_np):
            y=self.within_integral(x,lam)
            integrals[i]=np.trapz(y, x)
        
        self.agg_weight[:, 0] = torch.from_numpy(integrals).cuda()
        self.agg_weight[:,1]=1-self.agg_weight[:,0]
        for data in self.agg_weight[:, 0].cpu().view(-1).tolist():
    # 计算数据属于哪个区间
                index = int(data // self.bin_size)
    # 更新对应区间的计数器
                self.bin_count[index] += 1

        self.condition= self.div_vmf
        #self.condition=self.lams_np
        self.abstract_weight=copy.deepcopy(self.agg_weight.cpu())
        self.abstract_weight[:,1]=torch.from_numpy(self.condition/(1+self.condition))
        self.abstract_weight[:,0]=1-self.abstract_weight[:,1]
        self.abstract_entropy = -torch.sum(self.abstract_weight * torch.log(self.abstract_weight), dim=1)
      
        teacher_prob = F.softmax(g_out, dim=1)
        student_log_prob = F.log_softmax(p_out, dim=1)
        self.kl_g_p=F.kl_div(student_log_prob, teacher_prob, reduction='none').sum(1)
        self.kl_p_g=F.kl_div(F.log_softmax(g_out, dim=1), F.softmax(p_out, dim=1), reduction='none').sum(1)
        # beta 更新
        mask_alpha= (torch.Tensor(self.condition)<1)*(self.g_entropy.cpu()<self.global_en)*(self.l_entropy.cpu()>self.local_en) #*(self.abstract_entropy<self.mu)
        #mask_alpha= (torch.Tensor(1/(1+self.condition))) 
        delta_alpha=torch.sum(mask_alpha).item()
        mask_beta= (torch.Tensor(self.condition)>1) *(self.l_entropy.cpu()<self.local_en)*(self.g_entropy.cpu()>self.global_en)#*(self.abstract_entropy<self.mu)
        #mask_beta= 1-mask_alpha
        delta_beta=torch.sum(mask_beta).item()
        self.beta_alpha=self.beta_alpha+ delta_alpha
        self.beta_beta=self.beta_beta+ delta_beta
        restart_flag=self.beta_alpha+self.beta_beta
        if restart_flag>self.update_condition:
            self.beta_alpha=self.beta_alpha/restart_flag*self.update_condition/10+1
            self.beta_beta=self.beta_beta/restart_flag*self.update_condition/10+1
        self.beta_distribution = beta(self.beta_alpha, self.beta_beta)
        #return self.g_entropy,self.l_entropy,self.vmf_global_fz,self.vmf_local_fz
    
    def _validate_training(self, state, dataset,name):
        self.is_in_personalized_training = True
        # dont requires gradients.
        state["model"].requires_grad_(False)
        self.personal_head.requires_grad_(False)
        state["model"].eval()
        self.beta_alpha=1.0
        self.beta_beta=1.0

        self.bin_size = 0.05
        self.bin_count = [0] * int(1/self.bin_size)

        self.beta_distribution = beta(self.beta_alpha, self.beta_beta)
        tracker_te = RuntimeTracker(metrics_to_track=self.metrics.metric_names)
        for _input, _target in dataset:
            data_batch = create_dataset.load_data_batch(
                self.conf, _input, _target, is_training=False
            )
            g_out = state["model"](data_batch["input"])
            p_out = self.personal_head(self.rep_layer.rep)

            test_rep = self.rep_layer.rep.detach()
            temperature = torch.hstack((torch.ones((test_rep.shape[0], 1)).cuda(), torch.ones((test_rep.shape[0], 1)).cuda()))
            self.agg_weight = torch.nn.Parameter(torch.tensor(temperature).cuda(), requires_grad=False)
            
            self._calculate_samplewise_weight(test_rep,g_out,p_out)
            self._multi_head_inference(data_batch,g_out, p_out, tracker_te)
        
        
        self.logger.log("---------------------------------------------")
        self.logger.log(f"{name}")
        self.logger.log("vmf ratio")
        self.logger.log(f"{self.div_vmf[:10].tolist()}")
        self.logger.log("entropy ratio")
        self.logger.log(f"{self.div_entropy[:10].tolist()}")
        self.logger.log("alpha beta")
        self.logger.log(f"{self.beta_alpha}")
        self.logger.log(f"{self.beta_beta}")
        self.logger.log(".dim_vmf")
        self.logger.log(f"{self.dim_vmf[:10].tolist()}")
        self.logger.log("exp_entropy")
        self.logger.log(f"{self.pow_exp_local[:10]}")
        self.logger.log(f"{self.pow_exp_global[:10]}")
        self.logger.log("final lambda")
        self.logger.log(f"{self.lams_np[:10].tolist()}")
        self.logger.log(f"{self.agg_weight[:10,0].tolist()}")
        self.logger.log(f"{self.bin_count}")
        self.logger.log("\n")
        
        self.is_in_personalized_training = False
        self.agg_weight.data.fill_(1 / self.num_head)
        return tracker_te

    

    def _multi_head_inference(self,data_batch, g_out, p_out, tracker=None, g_pred=None, p_pred=None):
        # inference procedure for multi-head nets.
        y_g_origin=F.log_softmax(g_out, dim=1)
        y_l_origin=F.log_softmax(p_out, dim=1)
       
        self.l_entropy=self.l_entropy.unsqueeze(1)
        self.kl_p_g=self.kl_p_g.unsqueeze(1)
        self.g_entropy=self.g_entropy.unsqueeze(1)
        self.kl_g_p=self.kl_g_p.unsqueeze(1)
        
        fm_g=self.l_entropy+self.kl_p_g
    
        y_g=(self.l_entropy*y_g_origin+self.kl_p_g*y_l_origin)/fm_g
        fm_l=self.g_entropy+self.kl_g_p
        y_l=(self.g_entropy*y_l_origin+self.kl_g_p*y_g_origin)/fm_l
        y_g=y_g_origin *self.g_entropy/(self.l_entropy+self.g_entropy)
        y_l=y_l_origin *self.l_entropy/(self.l_entropy+self.g_entropy)
        agg_output = self.agg_weight[:, 0].unsqueeze(1) * y_g_origin  \
                         + self.agg_weight[:, 1].unsqueeze(1) *y_l_origin
        # evaluate the output and get the loss, performance.
        loss = self.criterion(agg_output, data_batch["target"])
        performance = self.metrics.evaluate(loss, agg_output, data_batch["target"])

        # update tracker.
        if tracker is not None:
            tracker.update_metrics(
                [loss.item()] + performance, n_samples=data_batch["input"].size(0)
            )

        return loss
    
    def get_rep(self):
        index=0
        self.global_rep_by_mean=np.zeros(self.conf.rep_len, dtype=np.float32)
        self.global_static_priors=[]
        self.global_vmf_priors=[]
        for i in range(self.conf.n_clients):        
            temp_covariance_matrix = torch.diag(self.global_rep_sigma[index:index+self.conf.rep_len])
            temp_numpy_covariance_matrix=temp_covariance_matrix.cpu().numpy()
            temp_mean=self.global_rep[index:index+self.conf.rep_len]
            temp_numpy_mean=temp_mean.cpu().numpy()
            self.global_rep_by_mean+=temp_numpy_mean/self.conf.n_clients
            self.global_static_priors.append(multivariate_normal(mean=temp_numpy_mean, cov=temp_numpy_covariance_matrix))
            self.global_vmf_priors.append(temp_numpy_mean)
            index+=self.conf.rep_len
        self.global_rep_by_mean=self.global_rep_by_mean*self.conf.n_clients/(self.conf.n_clients-1)- self.local_rep.cpu().numpy()/(self.conf.n_clients-1)

    def _validate(self, state, dataset_name):

        covariance_matrix = torch.diag(self.local_rep_sigma)
        numpy_covariance_matrix = covariance_matrix.cpu().numpy()
        mean=self.local_rep
        numpy_mean=mean.cpu().numpy()
        
        self.local_static_prior = multivariate_normal(mean=numpy_mean, cov=numpy_covariance_matrix)
        self.global_static_prior = multivariate_normal(mean=self.global_rep_by_mean, cov=numpy_covariance_matrix)
        
        # switch to evaluation mode.
        state["model"].eval()
        self.rep_layer = utils.determine_hook(state["model"], name=self.conf.arch)
        self.rep_layer.register_forward_hook(utils.hook)
        # test-time self-supervised aggregation
        tracker_te = self._validate_training(state, state[dataset_name],dataset_name)
        return tracker_te()

    def _get_target_histogram(self, display=True):
        local_data_loaders = self.fl_data_cls.intra_client_data_partition_and_create_dataloaders(
            localdata_id=self.client_id - 1,  # localdata_id starts from 0 while client_id starts from 1.
            other_ids=self._get_other_ids(),
            is_in_childworker=self.is_in_childworker,
            local_train_ratio=self.conf.local_train_ratio,
            batch_size=1,
            display_log=False,
        )
        hist = torch.zeros(utils.get_num_classes(self.conf.data))
        train_loader = local_data_loaders["train"]
        for _, _target in train_loader:
            hist[_target.item()] += 1
        if display:
            self.logger.log(
                f"\tWorker-{self.graph.worker_id} (client-{self.client_id}) training histogram is like {hist}"
            )
        return hist

   
    

    def _inference(self, data_batch, model, tracker=None):
        """Inference on the given model and get loss and accuracy."""
        # do the forward pass and get the output.
        
        output = model(data_batch["input"])
        
        rep=self.rep_layer_.rep 
        
        # evaluate the output and get the loss, performance.
        
         
        pos=self.cos(rep,self.local_rep.unsqueeze(0))
        logits = pos.reshape(-1, 1)
        for i in range(self.conf.n_clients):
            if i+1!=self.client_id:
                neg=self.cos(rep,torch.Tensor(self.global_vmf_priors[i]).unsqueeze(0).cuda())
                logits = torch.cat((logits, neg.reshape(-1, 1)), dim=1)
        logits/=self.conf.rep_len
        labels = torch.zeros(data_batch["input"].size(0)).to(self.graph.device).long()
       
        
        loss2 = self.nu * self.criterion(logits, labels)
        
        loss = self.criterion(output, data_batch["target"])#+loss2#-self.nu*self.cos(self.rep_layer.rep,self.local_rep.unsqueeze(0)).mean()
        performance = self.metrics.evaluate(loss, output, data_batch["target"])

        # update tracker.
        if tracker is not None:
            tracker.update_metrics(
                [loss.item()] + performance, n_samples=data_batch["input"].size(0)
            )
        return loss, output
    
    def train(self, model):
        # define dataloader, optimizer, scheduler and tracker
        state = self._init_train_process(model=model)
        self.rep_layer_ = utils.determine_hook(state["model"], name=self.conf.arch)
        self.rep_layer_.register_forward_hook(utils.hook)
        while not self._is_finished_one_comm_round(state):
            for _input, _target in state["train_loader"]:
                # load data
                with self.timer("load_data", epoch=state["scheduler"].epoch_):
                    data_batch = create_dataset.load_data_batch(
                        self.conf, _input, _target, is_training=True
                    )

                # inference and get current performance.
                with self.timer("forward_pass", epoch=state["scheduler"].epoch_):
                    state["optimizer"].zero_grad()
                    loss, _ = self._inference(
                        data_batch, state["model"], state["tracker"]
                    )
                    loss=loss+self.sigma*torch.norm(self.rep_layer_.rep,p=1)/32
                with self.timer("backward_pass", epoch=state["scheduler"].epoch_):
                    loss.backward()
                    state["optimizer"].step()
                    state["scheduler"].step()

                if self.conf.display_log:
                    self._display_logging(state)
                if self._is_diverge(state):
                    self._terminate_comm_round(state)
                    return state  
            # refresh the logging cache at the end of each epoch.
            state["tracker"].reset()
            if self.logger.meet_cache_limit():
                self.logger.save_json()

        # terminate
        self._terminate_comm_round(state)

        return state
    
    @torch.no_grad()
    def _pro_train(self,model):
        state = self._init_train_process(model=model)
        
        self.rep_layer = utils.determine_hook(state["model"], name=self.conf.arch)
        self.rep_layer.register_forward_hook(utils.hook)
        matrix=torch.rand(0, self.conf.rep_len).to(self.graph.device)
        for e in range(1):
            for _input, _target in state["train_loader"]:
                data_batch = create_dataset.load_data_batch(
                        self.conf, _input, _target, is_training=True,
                    )
                _= state["model"](data_batch["input"])
                input_real=self.rep_layer.rep 
                matrix= torch.cat([matrix, self.fsq_quantize(input_real,self.q_level)], dim=0)
        empirical_distributions = torch.stack([torch.histc(data.float(), bins=self.q_level, min=0, max=self.q_level-1) /len(data)  for data in matrix.t()], dim=1)   
        self.pro.category_parameters = nn.Parameter(empirical_distributions) 
        print(self.client_id)
        print(self.pro.category_parameters[:,:10])
    
    def send_pro(self):
        self.pro=self.pro.cpu()
        dist.barrier()
        head_dict = copy.deepcopy(self.pro.state_dict())
        head_values=list(head_dict.values())
        flatten_head=TensorBuffer(head_values)
        dist.send(tensor=flatten_head.buffer,dst=0)
        dist.barrier()
        self.pro= self.pro.to(self.graph.device)
    
    def fsq_quantize(self,tensor, num_values):
        #sigmoid_output = torch.sigmoid(tensor)
        tanh_output =F.tanh(tensor)
        normalized = tanh_output * (num_values - 1)
        quantized_output = torch.round(normalized)
        return (quantized_output-normalized).detach()+normalized
    
    def _recv_pro_from_master(self):
        #old_buffer = copy.deepcopy(self.model_tb.buffer)
        self.global_pro= CategoryDistributionModel(self.conf.rep_len,self.q_level)
        state_dict=self.global_pro.state_dict()
        model_tb = TensorBuffer(list(state_dict.values()))
        dist.recv(model_tb.buffer, src=0)
        model_tb.unpack( state_dict.values())
        self.global_pro.load_state_dict(state_dict)
        self.global_pro= self.global_pro.to(self.graph.device)
        dist.barrier()
        with torch.no_grad():
            self.global_pro.category_parameters.data=self.global_pro.category_parameters.data*self.conf.n_clients/(self.conf.n_clients-1)-self.pro.category_parameters.data/(self.conf.n_clients-1)
    
    def _recv_pros_from_master(self):
        #old_buffer = copy.deepcopy(self.model_tb.buffer)
        self.pro_models=dict()
        for selected_client_id in range(1,self.conf.n_clients+1):
            self.pro_models[selected_client_id]=CategoryDistributionModel(self.conf.rep_len,self.q_level)
        for i in range(1,self.conf.n_clients+1):
            state_dict=self.pro_models[i].state_dict()
            model_tb = TensorBuffer(list(state_dict.values()))
            dist.recv(model_tb.buffer, src=0)
            model_tb.unpack(state_dict.values())
            self.pro_models[i].load_state_dict(state_dict)
            self.pro_models[i]= self.pro_models[i].to(self.graph.device)
            dist.barrier()

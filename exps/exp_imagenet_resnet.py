import itertools


class NewConf(object):
    # create the list of hyper-parameters to be replaced.
    to_be_replaced = dict(
        # general for world.
        experiment=["debug"],
        # use world to control the distribution of clients on cuda devices.
        # for advanced usage, use world_conf instead, see ./pcode/utils/topology.py
        world=["0,0,0,0,0,0,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2"],
        on_cuda=[True],
        python_path=["/opt/conda/bin/python"],
        hostfile=["env/hostfile"],
        manual_seed=[7],
        same_seed_process=[False],
        # general for the training.
        track_time=[True],
        display_tracked_time=[True],
        # general for fl.
       
        data=["imagenet"],
        natural_shifted_imagenet_type=["imagenet_v2_matched-frequency"],  # useless
        data_dir=["/home/default_user/data"],
        batch_size=[128],
        num_workers=[0],
        # fl master
        n_comm_rounds=[90],
        early_stopping_rounds=[0],
        # fl clients
        group_norm_num_groups=[2],
        rep_len=[256],
        arch=["resnet20"],
        complex_arch=["master=resnet20,worker=resnet20"],
        optimizer=["sgd"],
        momentum_factor=[0.9],
        # likely to be changed
        local_n_epochs=[5],
        n_personalized_epochs=[1],
        lr=[0.01],
        personal_lr=[0.01],
        q_level=[2],
        sigma=[0.0001],
        mu=[0.7],
        nu=[0],
        cof=[5],
        temperature=[0.05],
        kb=[4],
       
        update_condition=[300],
        
        talpha=[0.5],
        participation_ratio=[1.0],
        partition_data_conf=["distribution=non_iid_dirichlet,non_iid_alpha=1.0,size_conf=1:1"],
        personalization_scheme=[  
                                "method=THE",
                               "method=BTFL", 
                                "method=Memo_personal", 
                                "method=Fine_tune",                                                      
                               "method=Normal", 
                               "method=knn_per", 
                                 "method=FedRep", 
                                   "method=FedRod", 
                                     "method=ttt",                                 
                               ],
    )

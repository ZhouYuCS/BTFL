import itertools


class NewConf(object):
    # create the list of hyper-parameters to be replaced.
    to_be_replaced = dict(
        # general for world.
        experiment=["debug"],
        # use world to control the distribution of clients on cuda devices.
        # for advanced usage, use world_conf instead, see ./pcode/utils/topology.py
        world=["0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"],
        on_cuda=[True],
        python_path=["/opt/conda/bin/python"],
        hostfile=["env/hostfile"],
        manual_seed=[7],
        same_seed_process=[False],
        # general for the training.
        track_time=[True],
        display_tracked_time=[True],
        # general for fl.
        n_clients=[6],
        data=["oh"],
        data_dir=["/home/default_user/data"],
        batch_size=[32],
        num_workers=[0],
        # fl master
        n_comm_rounds=[40],
        rep_len=[512],
        early_stopping_rounds=[0],
        # fl clients
        group_norm_num_groups=[2],
        arch=["resnet18"],
        complex_arch=["master=resnet18,worker=resnet18"],
        optimizer=["sgd"],
        momentum_factor=[0.9],
        local_n_epochs=[5],
        n_personalized_epochs=[1],
        lr=[0.01],
        personal_lr=[0.01],
        participation_ratio=[1.0],
        q_level=[2],
        sigma=[0],
        mu=[0.7],
        nu=[0],
        cof=[5],
        update_condition=[50],
        temperature=[0.05],
        kb=[4],
     
       
        talpha=[0.5],
        partition_data_conf=["distribution=non_iid_dirichlet,non_iid_alpha=0.1,size_conf=1:1",
        ],
        personalization_scheme=[
                            
                                 "method=Memo_personal",
                                 
                                                                      
                                ],
    )
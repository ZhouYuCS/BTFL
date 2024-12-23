import itertools


class NewConf(object):
    # create the list of hyper-parameters to be replaced.
    to_be_replaced = dict(
        # general for world.
        experiment=["debug"],
        # use world to control the distribution of clients on cuda devices.
        world=["0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1"],
        on_cuda=[True],
        python_path=["/opt/conda/bin/python"],
        hostfile=["env/hostfile"],
        manual_seed=[7],
        same_seed_process=[False],
        # general for the training.
        track_time=[True],
        display_tracked_time=[True],
        # general for fl.
        n_clients=[20],
        #n_master_sampled_clients=[10],
        data=["cifar10"],
        data_dir=["/home/default_user/data"],
        batch_size=[32],
        num_workers=[0],
        # fl master
        n_comm_rounds=[80],
        early_stopping_rounds=[0],
        # fl clients
        rep_len=[128],
        arch=["compact_conv_transformer"],
        complex_arch=["master=compact_conv_transformer,worker=compact_conv_transformer"],
        optimizer=["adam"],
        local_n_epochs=[5],
        n_personalized_epochs=[1],
        q_level=[2],
        sigma=[0.0001],
        mu=[0.7],
        nu=[0],
        cof=[5],
        temperature=[0.05],
        kb=[4],
       
        update_condition=[300],
        talpha=[0.5],
        lr=[0.001],
        personal_lr=[0.001],
        participation_ratio=[1.0],
        partition_data_conf=["distribution=non_iid_dirichlet,non_iid_alpha=0.1,size_conf=1:1"],
        personalization_scheme=[     
                           "method=Memo_personal",
                               ],              
    )

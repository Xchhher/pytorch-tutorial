import random
import numpy as np
import torch
import os

# global random seed
def seed_everything(seed = 20):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置dataloader的种子 
def worker_init_fn(worker_id,rank,seed):
    worker_seed = rank + seed
    random.seed(worker_id)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

# 断点续训 这里是指定了路径 移植到其他文件的时候需要对路径以及名称进行修改
def save_model(model,optimizer,num_epochs):
        # Save the model checkpoint 训练的过程中保存模型
    checkpoint = {
        "net":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        "epoch":num_epochs
    }
    if not os.path.isdir("tutorials/01-basics/feedforward_neural_network/models/checkpoint"):
        os.mkdir("tutorials/01-basics/feedforward_neural_network/models/checkpoint")
    torch.save(checkpoint, 'tutorials/01-basics/feedforward_neural_network/models/checkpoint/ckpt_best_%s.pth' %(str(num_epochs)))


# 断点继续训练 针对 学习率不变
def reload_model(model,optimizer,epoch,RESUME=False):
    # epoch的恢复
    start_epoch = -1

    if RESUME:
        path_checkpoint = f"tutorials/01-basics/feedforward_neural_network/models/checkpoint/ckpt_best_{epoch}.pth"  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点

        model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch

    return start_epoch


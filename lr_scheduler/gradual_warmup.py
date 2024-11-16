from torch.optim.lr_scheduler import  _LRScheduler
from config import config
from logger import logger

class  GradualWarmupScheduler(_LRScheduler):
    """
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, start_warmup_lr , total_epoch, after_scheduler=None):

        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished =  False

        self.start_warmup_lr = start_warmup_lr
        super().__init__(optimizer)

    def get_lr(self):
        logger.debug(f"GradualWarmupScheduler last_epoch : {self.last_epoch}")
        if self.last_epoch >= self.total_epoch:
            if self.after_scheduler:
                if  not self.finished:
                    self.after_scheduler.base_lrs =  [ base_lr for base_lr in self.base_lrs]
                    self.finished =  True
                return self.after_scheduler.get_lr()
            return  [base_lr for base_lr in self.base_lrs]
        return  [ ((base_lr - self.start_warmup_lr )  * (self.last_epoch) / (self.total_epoch-1) +  self.start_warmup_lr)  for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            return self.after_scheduler.step(epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)
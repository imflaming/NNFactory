from torch.optim.lr_scheduler import  _LRScheduler
from logger import logger
from torch.optim.lr_scheduler import StepLR , ExponentialLR


class  PhaseReductionScheduler(_LRScheduler):
    """
    阶段下降 phase： dict  =  {100 : 0.01 , 150 : 0.001}
    """

    def __init__(self, optimizer, phase: dict, total_epochs : int, start_epoch : int = None):
        


        self.phase = phase
        self.total_epochs = total_epochs
        self.points : list  = list(self.phase.keys())
        
        self.points.sort()
        self.phase[0] = self.phase[self.points[0]]
        self.points.insert(0 , 0)
        logger.debug(f"self.points {self.points}")
        logger.debug(f"self.phase {self.phase}")


        super().__init__(optimizer)

        if start_epoch:
            self.last_epoch = start_epoch

        

    def get_lr(self):
        logger.debug(f"PhaseReductionScheduler last_epoch : {self.last_epoch}")
        for idx , item in enumerate(self.points[1:] ,start=1):
            if self.last_epoch >= self.points[idx-1] and self.last_epoch < item: 
                return [ self.phase.get( item ) for base_lrs in self.base_lrs ]
        return [ self.phase.get(  self.points[-1]) for base_lrs in self.base_lrs ]
    
    def step(self, epoch = None):
        logger.debug(f"PhaseReductionScheduler step")
        return super(PhaseReductionScheduler, self).step(epoch)

    
        

            
    


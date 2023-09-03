import tensorflow as tf
import numpy as np


class splitter():
    '''
    creates 3 splits
    train
    dev
    test
    '''
    def __init__(self, tasker, train_proportion = 0.7, dev_proportion = 0.1):
        self.train_proportion = train_proportion
        self.dev_proportion = dev_proportion
        ## These could be input args in the future
        self.num_hist_steps = 10
        self.task = 'link_pred'
        #only the training one requires special handling on start, the others are fine with the split IDX.
        start = int(tasker.data.min_time + self.num_hist_steps)
        end = self.train_proportion
        end = int(np.floor(tf.cast(tasker.data.max_time+1, tf.float64) * end))
        train = data_split(tasker, start, end, test = False)

        start = end
        end = self.dev_proportion + self.train_proportion
        end = int(np.floor(tf.cast(tasker.data.max_time+1, tf.float64) * end))
        dev = data_split(tasker, start, end, test = True)    

        start = end
        end = int(tasker.max_time) + 1
        test = data_split(tasker, start, end, test = True)
             
        print ('Dataset splits sizes:  train',len(train), 'dev',len(dev), 'test',len(test))
        
        self.tasker = tasker
        self.train = train
        self.dev = dev
        self.test = test
        


class data_split:
    def __init__(self, tasker, start, end, test, **kwargs):
        '''
        start and end are indices indicating what items belong to this split
        '''
        self.tasker = tasker
        self.start = start
        self.end = end
        self.test = test
        self.kwargs = kwargs
    def __len__(self):
        return self.end-self.start
    def __getitem__(self,idx):
        if idx < 0 or idx >= self.__len__():
            raise IndexError("Index out of bounds")
        t = self.tasker.get_sample(self.start + idx)
        return t
    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


'''
Created on Jul 12, 2018

@author: loitg
'''

class Config(object):
    out_charset = ''
    channels = 1 #3
    
    layer_params = [ [  64, 3, 'valid', 'conv1', False],
                 [  64, 3, 'same',  'conv2', True], # pool
                 [ 128, 3, 'same',  'conv3', False],
                 [ 128, 3, 'same',  'conv4', True], # hpool
                 [ 256, 3, 'same',  'conv5', False],
                 [ 256, 3, 'same',  'conv6', True], # hpool
                 [ 512, 3, 'same',  'conv7', False],
                 [ 512, 3, 'same',  'conv8', True]] # hpool 3

    rnn_size1 = 2**9
    rnn_size2 = 2**9
    


class CMND9SoConfig(Config):
    out_charset = '0123456789'
    channels = 1 #3

    rnn_size1 = 2**9
    rnn_size2 = 2**9
    
class CMND9ChuConfig(Config):
    out_charset = '0123456789'
    channels = 1 #3

    rnn_size1 = 2**9
    rnn_size2 = 2**9


if __name__ == '__main__':
    pass
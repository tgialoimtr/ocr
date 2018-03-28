'''
Created on Feb 19, 2018

@author: loitg
'''
# This is a placeholder for a Google-internal import.
from time import time
from Queue import Empty, Full
from common import args
    
class BatchLinePredictor(object):
    def __init__(self, server, logger):
        self.putq = server.queue_get
        self.getq = server.queue_push
        
    def predict_batch(self, batch_name, img_list, logger):
        self.putq.put((batch_name, img_list), block=True)
        logger.debug('put %d imgs to queue put', len(img_list))
        pred = {}
        waitcount = 0
        while True:
            try:
                if waitcount > args.qget_wait_count:
                    logger.warning('WAITING SERVER TOO LONG ...') 
                topqueue = self.getq.get(timeout=args.qget_wait_interval)
                batchid, texts = topqueue
                if batchid != batch_name:
                    waitcount += 1
                    self.getq.put((batchid, texts)) 
                    continue
                else:
                    for i, txt in enumerate(texts):
                        pred[i] = txt
                    return pred
            except Empty:
                waitcount += 1       
        
#         for i, img in enumerate(img_list):
#             self.putq.put((batch_name + '_' + str(i), time(), img), block=True)
#         logger.debug('put %d imgs to queue put %s', len(img_list), self.clientid)
#         pred = {}
#         waitcount = 0
#         while True:
#             try:
#                 topqueue = self.getq.get(timeout=args.qget_wait_interval)
#                 imgid, txt = topqueue
#                 [batchid, imgid] = imgid.rsplit('_',1)
#                 if batchid != batch_name: continue
#                 pred[int(imgid)] = txt
#                 if len(pred) == len(img_list):
#                     return pred
#             except Empty:
#                 waitcount += 1
#                 if waitcount > args.qget_wait_count:
#                     logger.warning('WAITING SERVER TOO LONG ...')
                
                            
    
if __name__ == '__main__':
    pass
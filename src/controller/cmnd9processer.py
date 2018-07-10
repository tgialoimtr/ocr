'''
Created on Jul 10, 2018

@author: loitg
'''
import cv2
import time
import multiprocessing
from multiprocessing import Process, Manager, Pool

from extras.weinman.interface import server_chu as LocalServer_chu
from extras.weinman.interface import server_so as LocalServer_so
from extras.weinman.interface import linepredictor
from utils.common import createLogger, args

def runserver(server, states):
    logger = createLogger('server')
    server.run(states, logger) 
    
class Cmnd9Processer(object):


    def __init__(self, bzid):
        self.cancuoc_tmpl = StaticTemplate.createFrom('/home/loitg/Downloads/cmnd_data/template/1cmnd9.xml')
    
    
    def init(self):
        logger = createLogger('reader')
        manager = Manager()
        states = manager.dict()
        server0 = LocalServer_so(args.model_path, manager)
        print str(server0.queue_get)
        server1 = LocalServer_chu(args.model_path_chu, manager)
        print str(server1.queue_get)
        p = Process(target=runserver, args=(server0, states))
        p_chu = Process(target=runserver, args=(server1, states))
        p.daemon = True
        p_chu.daemon = True
        p.start()
        p_chu.start()
        return 'started: ' + str(p.pid) + ' and ' + str(p_chu.pid)
    
    ### This is NOT multi-thread safe
    def process1(self, filepath):
        img = cv2.imread(filepath)
        prob, lines = self.cancuoc_tmpl.find(img)
        if prob > 0.5:
            print prob
            for k, imgline in lines.iteritems():
                print k
                cv2.imshow('line', imgline.img)
                cv2.waitKey(-1)

        line = cv2.cvtColor(line, cv2.COLOR_BGR2RGB)
        linereader = BatchLinePredictor(server, logger)
    
        newwidth = int(32.0/line.shape[0] * line.shape[1])
        if newwidth < 32 or newwidth > common.args.stdwidth: return 'Line too short or long'
        line = cv2.resize(line, (newwidth, 32))
        for i in range(args.batch_size):
            line_list.append(line)
            
        batchname = datetime.datetime.now().isoformat()
        pred_dict = linereader.predict_batch(batchname, line_list, logger)
        rs = pred_dict[0] if len(pred_dict[0].strip()) > 0 else '<blank line>'
        return rs 



    ### This is multi-thread safe
    def read(self, filepath):
        img = cv2.imread(filepath)     
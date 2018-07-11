'''
Created on Jul 10, 2018

@author: loitg
'''
import os
import cv2
import datetime
from multiprocessing import Process, Manager

from extras.weinman.interface.server_chu import LocalServer as LocalServer_chu
from extras.weinman.interface.server_so import LocalServer as LocalServer_so
from extras.weinman.interface.linepredictor import BatchLinePredictor
from utils.common import createLogger, args
from cmnd.template import StaticTemplate

def runserver(server, states):
    logger = createLogger('server')
    server.run(states, logger) 
    
class Cmnd9Processer(object):


    def __init__(self, bzid, logger=None):
        if logger is None: logger = createLogger('bzid')
        self.logger = logger
        self.cancuoc_tmpl = StaticTemplate.createFrom('/home/loitg/Downloads/cmnd_data/template/1cmnd9.xml')
    
    
    def init(self):
        manager = Manager()
        states = manager.dict()
        self.server_so = LocalServer_so(args.model_path_so, manager)
        print str(self.server_so.queue_get)
        self.server_chu = LocalServer_chu(args.model_path_chu, manager)
        print str(self.server_chu.queue_get)
        p_so = Process(target=runserver, args=(self.server_so, states))
        p_chu = Process(target=runserver, args=(self.server_chu, states))
        p_so.daemon = True
        p_chu.daemon = True
        p_so.start()
        p_chu.start()
        return 'started: ' + str(p_so.pid) + ' and ' + str(p_chu.pid)
    
    ### This is NOT multi-thread safe
    def process1(self, filepath):
        img = cv2.imread(filepath)
        prob, lines = self.cancuoc_tmpl.find(img)
        rs = {}
        if prob > 0.5:
            print prob
            reader_chu = BatchLinePredictor(self.server_chu, self.logger)
            list_chu = None*4
            reader_so = BatchLinePredictor(self.server_so, self.logger)
            list_so = None*4
            for k, imgline in lines.iteritems():
                if k in ['whole', 'thuongtru1', 'thuongtru2']: continue
                line = cv2.cvtColor(imgline.img, cv2.COLOR_BGR2RGB)
                newwidth = int(32.0/line.shape[0] * line.shape[1])
                if newwidth < 32 or newwidth > args.stdwidth: return 'Line too short or long'
                line = cv2.resize(line, (newwidth, 32))
                if k == 'id':
                    list_so[0] = line
                    list_so[2] = line*0.85
                if k == 'ntns':
                    list_so[1] = line
                    list_so[3] = line*0.85
                elif k == 'hoten1':
                    list_chu[0] = line
                elif k == 'hoten2':
                    list_chu[1] = line                    
                elif k == 'quequan1':
                    list_chu[2] = line
                elif k == 'quequan2':
                    list_chu[3] = line
            batchname = datetime.datetime.now().isoformat()
            pred_dict_chu = reader_chu.predict_batch(batchname, list_chu, self.logger)
            pred_dict_so = reader_so.predict_batch(batchname, list_so, self.logger)
            rs['id'] = pred_dict_so[0]
            rs['ntns'] = pred_dict_so[1]
            rs['hoten'] = pred_dict_chu[0].strip() + pred_dict_chu[1].strip()
            rs['quequan'] = pred_dict_chu[2].strip() + pred_dict_chu[3].strip()
        return rs 



    ### This is multi-thread safe
    def read(self, filepath):
        return self.process1(filepath)
    
    
if __name__ == '__main__':
    p = Cmnd9Processer('cmnd9')
    p.init()
    pathp = '/home/loitg/workspace/clocr/temp/1/'
    for fn in os.listdir(pathp):
        a = p.read(pathp + fn)
        print a


 

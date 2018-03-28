'''
Created on Mar 28, 2018

@author: loitg
'''
import time
import multiprocessing
from multiprocessing import Process, Manager, Pool
import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
import logging
from logging.handlers import TimedRotatingFileHandler

from common import args
from server import LocalServer
from pagepredictor import PagePredictor
from extract_fields.extract import CLExtractor
from receipt import ReceiptSerialize, ExtractedData

app = Flask(__name__)
manager = None
server = None
logger = None

def createLogger(name):
    logFormatter = logging.Formatter("%(asctime)s [%(name)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger(name)
    
    fileHandler = TimedRotatingFileHandler(os.path.join(args.logsdir, 'log.' + name) , when='midnight', backupCount=10)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.DEBUG)
    return rootLogger

def runserver(server, states):
    logger = createLogger('server')
    server.run(states, logger) 


@app.route('/start')
def start():
    global manager,server, logger
    logger = createLogger('reader')
    manager = Manager()
    states = manager.dict()
    server = LocalServer(args.model_path, manager)
    p = Process(target=runserver, args=(server, states))
    p.daemon = True
    p.start()
    return 'started: ' + str(p.pid)


def read(imgpath, logger):
    try:
        reader = PagePredictor(server, logger)
        extractor = CLExtractor()
        lines = reader.ocrImage(imgpath, logger)
        extdata = extractor.extract(lines)
    except Exception:
        logger.exception('Exception')
        extdata = ExtractedData()
    rinfo = ReceiptSerialize()
    outmsg = rinfo.combineExtractedData(extdata)
    return outmsg


@app.route('/ocr/read', methods=['GET', 'POST'])
def upload_file():
    global logger
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file0 = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file0.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file0:
            filename = secure_filename(file0.filename)
            imgpath = os.path.join(args.download_dir, filename)
            file0.save(imgpath)
            outmsg = read(imgpath, logger)
            os.remove(imgpath)
            return outmsg
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''
            
if __name__ == '__main__':
    app.run(port=int("8014"))
    
    
    
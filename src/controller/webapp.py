'''
Created on Mar 28, 2018

@author: loitg
'''
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from utils.common import createLogger
from controller.cmnd9processer import Cmnd9Processer
from common import args

app = Flask(__name__)
mtreader = None

import datetime, random
def makeNameUnique(rawName):
    newName =  datetime.datetime.now().isoformat() \
                 + str(random.randint(1,99)) + '_' + rawName
    return newName

@app.route('/ocr/cmnd9/', methods=['GET', 'POST'])
def upload_file():
    ### Init variables
    global mtreader, logger
    bzid = 'cmnd9'
    processer_cls = Cmnd9Processer
    ### start function
    if mtreader is None:
        logger = createLogger(bzid, args.logsdir)
        mtreader = processer_cls(bzid, logger)
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
            filename = makeNameUnique(secure_filename(file0.filename))
            imgpath = os.path.join(args.download_dir, filename)
            file0.save(imgpath)
            outmsg = mtreader.read(imgpath)
#             os.remove(imgpath)
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
    app.run(port=int("8080"))
    
    
    
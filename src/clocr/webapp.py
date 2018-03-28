'''
Created on Mar 28, 2018

@author: loitg
'''
import time
import multiprocessing
import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
backProc = None
UPLOAD_FOLDER = '/tmp/uploads/'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def testFun():
    print('Starting')
    while True:
        time.sleep(3)
        print('looping')
        time.sleep(3)
        print('3 Seconds Later')

@app.route('/')
def root():

    return 'Started a background process with PID111 ' + str(backProc.pid) + " is running: " + str(backProc.is_alive())

@app.route('/kill')
def kill():
    backProc.terminate()
    return 'killed: ' + str(backProc.pid)

@app.route('/kill_all')
def kill_all():
    proc = multiprocessing.active_children()
    for p in proc:
        p.terminate()
    return 'killed all'

@app.route('/active')
def active():
    proc = multiprocessing.active_children()
    arr = []
    for p in proc:
        print(p.pid)
        arr.append(p.pid)

    return str(arr)

@app.route('/start')
def start():
    global backProc
    backProc = multiprocessing.Process(target=testFun, args=())
    backProc.daemon = True
    backProc.start()
    return 'started: ' + str(backProc.pid)

@app.route('/file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
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
    app.run(port=int("7879"))
    
    
    
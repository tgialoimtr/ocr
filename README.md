# ocr
In future, this will be the prediction package. For now, it only has line image extraction module.    

# install dependencies:
install opencv contrib modules (sift, orb, ...)   
pip install ...   

# Run:
* import project to Eclipse + Pydev   
* edit hard-coded paths /home/.../cmnd_data/... in codes (using text finder of Eclipse) to actual cmnd_data folder on disk   
* In src/clocr/main2.py: edit cmnd folder (variable "cmnd_path") to where contains cmnd images to extract lines (samples are cmnd_data/realcapture)
* Run this python file in Eclipse to see result line image extracted.

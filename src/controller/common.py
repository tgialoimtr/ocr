'''
Common arguments and parameters
'''
class DFO(object):
    pass

args = DFO()
args.model_path = '/home/loitg/debugtf/model_version4_total/'#'/home/loitg/location_nn/tfmodel/'
args.imgsdir = '/home/loitg/ctoaia/case1_line/hard_imgs/'
args.numprocess = 1 #CPU:2 #GPU:8
args.qget_wait_count = 400000
args.qget_wait_interval = 0.3
args.stdwidth=32*20
args.batch_size = 16
args.bucket_size = 1 #CPU:2 #GPU:16 
args.bucket_max_time = 10
args.device = '/device:CPU:0'
args.javapath = '/home/loitg/location_nn/java'
args.logsdir = '/home/loitg/location_nn/logs/'
args.dbfile = 'top200_v3.csv'
args.locationnjar = 'location_nn.jar'
args.download_dir = '/home/loitg/location_nn/downloads'
args.connection_string = 'DefaultEndpointsProtocol=http;AccountName=storacctcapitastartable;AccountKey=Z/dhpkNhR7DY0goHVsaPldFCnqzydIN/CunYh324E8M82eqOGeupYFS5CGz7CS18FDm1wWmWPEX3ecxJ23HqmA=='
args.queue_get_name = 'ocr-receipt-queue-hangtest'
args.queue_push_name = 'receipt-info-queue-hangtest'
args.container_name = 'mobile-receipts-dev-hangtest'
args.receipt_waiting_interval = 10
args.heartbeat_check = 300
args.mode = 'process'
##########################
args = DFO()
args.model_path = '/home/loitg/workspace/poc_aia_resources/model_id-so/'
args.model_path_chu = '/home/loitg/workspace/poc_aia_resources/model_chu3/'
args.imgsdir = '/home/loitg/ctoaia/case1_line/hard_imgs/' # test image for validate.py
args.numprocess = 1 #CPU:2 #GPU:8
args.qget_wait_count = 400000
args.qget_wait_interval = 0.3
args.stdwidth=32*20
args.bucket_size = 1 #CPU:2 #GPU:16 
args.batch_size = 2
args.bucket_max_time = 10
args.device = '/device:CPU:0'
args.logsdir = '/home/loitg/location_nn/logs/' # store logs files
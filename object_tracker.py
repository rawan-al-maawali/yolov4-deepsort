################Import Libraries#################
import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#to calculate FPS, time, and date
import time 
from datetime import datetime, date 
cdate = date.today()
ctime =datetime.now().strftime("%H:%M")


import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
#initializing the flag settings for the terminal
from absl import app, flags, logging
from absl.flags import FLAGS 


import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image


import cv2 #to visualize the tracking

import numpy as np


import matplotlib.pyplot as plt # for the color map


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from _collections import deque


## deep sort imports ##   
from deep_sort import preprocessing        #used for the non_max_suppression
from deep_sort import nn_matching          # setting up deep associations
from deep_sort.detection import Detection  # for detecting an object
from deep_sort.tracker import Tracker      # for writing the tracking info


###########Database imports #############
##uncomment when integrated with SITL+database+dashboard
#import pymongo
#from pymavlink import mavutil

from tools import generate_detections as gdet # import the feature generation encoder



#######Define Yolo models and load the weights into the models ######
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')


##################MongDB Connection#####################
##uncomment when integrated with SITL+database+dashboard
# client = pymongo.MongoClient("mongodb+srv://admin:admin123@cluster0.ddalf.mongodb.net/?retryWrites=true&w=majority")
# database = client["WildObs_Detection_f"]

#############Initiating object connection with DB#######
##uncomment when integrated with SITL+database+dashboard
# collection = database["Horse"]
# collection_horse_eating = database["Horse_eating"]
# collection_horse_sleeping = database["Horse_sleeping"]
# horse_ID = set()
# horse_eating_ID = set()
# horse_sleeping_ID = set()

######## port udp 14550 with mavproxy#########
##uncomment when integrated with SITL+database+dashboard
# the_connection = mavutil.mavlink_connection('udpin:localhost:14551')
# the_connection.wait_heartbeat()



def main(_argv):
    ############### Definition of the parameters###########
    max_cosine_distance = 0.4 # Used to consider if the object is the same or not; if the cosine distance is bigger than
    # 0.5, it means the features are very similar in the object in previous frame and object in current frame
    nn_budget = None  # used to create libraries and store the features letters - which are extracted using deep network,
    # by default this is 100, but now it is not enforced (None)  
    nms_max_overlap = 1.0  # to avoid if there are too many detections for the same object. By default, this is 1 (keep
    # the old detection), but it is not a good idea because there might be too many detections for the same object
    
    ########### initialize deep sort functions##############
    model_filename = 'model_data/mars-small128.pb' # pre-trained convolution neural network for tracking pedestrians
    encoder = gdet.create_box_encoder(model_filename, batch_size=1) # feature generations
    # calculate cosine distance association metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric) # Pass the association matrix to the DeepSort Tracker

    ###############load configuration for object detector############
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    ############### begin video capture ##############
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None
    ##############Output#####################
    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    pts = [deque(maxlen=30) for _ in range(1000)]
    
    #Empty lists for counting the horses
    counter_horse = []
    counter_sleeping = []
    counter_eating = []
    
    ########### while video is running###############
    while True:
        return_value, frame = vid.read()
     
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # The color in opencv is BGR, but in Tensorflow is RGB, so we need
            # to convert it
            image = Image.fromarray(frame)
            cv2.rectangle(frame, (0,0), (1920,50), (0,0,255), -1) #draw rectangle for the title
            cv2.rectangle(frame, (0,50), (1920,100), (0,0,0), -1) #draw rectangle for FPS and Counting info
            cv2.rectangle(frame,(0, 1030) , (1920, 1080), (0,0,0), -1) #draw rectangle for time and date info

            
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size)) # resizing our image
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]
     
         # this information will be used to perform long maximum suppression on the 
         #detection frame to eliminate multiple frames on one target
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0] 
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map, which maps numbers to colors
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)] # 20 color list

        ############### run non-maxima supression###############
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]  # To remove the redundancy of boxes     

        ########### Call the tracker ###########################
        tracker.predict() # To propagate the track distribution one time step forward - based on Kilman Filtering
        tracker.update(detections)


        current_horse_count = int(0)
        current_horse_eating_count = int(0)
        current_horse_sleeping_count = int(0)
        
        

  
        ############# update tracks#########################
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue  # If there is no kilman filtering in the track or no update, skip the track
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
            
            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            # Another box for tracking id
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            
            
 
            ############# Historical trajectory path ####################
            #uncomment this for historircal trajectory path
            #This path follows the horse to indicate its path history as a line
            # center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
            # pts[track.track_id].append(center)  # Use queues to store all center points

            # for j in range(1, len(pts[track.track_id])):
                # if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                    # continue
                # thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                # cv2.line(frame, (pts[track.track_id][j - 1]), (pts[track.track_id][j]), color, thickness)
  
            ###########Checking class name and counting horses ############
            if class_name == 'Horse'or class_name == 'Horse_eating' or class_name == 'Horse_eating':
                counter_horse.append(int(track.track_id))
                current_horse_count += 1
                #uncomment below line when integrated with SITL+database+dashboard
                #horse_ID.add(class_name +' '+ str(track.track_id))
            if class_name == 'Horse_sleeping':
                counter_sleeping.append(int(track.track_id))
                current_horse_sleeping_count += 1
                #uncomment below line when integrated with SITL+database+dashboard
                #horse_sleeping_ID.add(class_name +' '+ str(track.track_id))
            if class_name == 'Horse_eating':
                counter_eating.append(int(track.track_id))
                current_horse_eating_count += 1
                ##uncomment below line when integrated with SITL+database+dashboard
                #horse_eating_ID.add(class_name +' '+ str(track.track_id))

        fps = 1.0 / (time.time() - start_time) #calculating FPS 
        cv2.putText(frame, "Horses Tracking System" , (900, 25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255,255,255), 1)
        
        
        #########Writing Information on screen ############################
        cv2.putText(frame, "Horses being tracked:" + str(current_horse_count), (480, 75), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255,255,255), 1)
        cv2.putText(frame, "Horses Eating: " + str(current_horse_eating_count), (960, 75), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255,255,255), 1)
        cv2.putText(frame, "Horses Sleeping: " + str(current_horse_sleeping_count), (1440, 75), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255,255,255), 1)
        cv2.putText(frame, "Time: " + datetime.now().strftime("%H:%M"), (900, 1050), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255,255,255), 1)
        cv2.putText(frame, ("Video Source:" + video_path), (10, 1050), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255,255,255), 1)
        cv2.putText(frame, ("Date: " + str(cdate)), (1700, 1050), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255,255,255), 1)
        cv2.putText(frame, "FPS: %.2f" % fps ,(10, 75), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (255,255,255),1)
     
        
        

        # if enable info flag then print details about each track
        if FLAGS.info:
            print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        

        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()
    
    
    
    #uncomment below lines when integrated with SITL+database+dashboard
    #####Posting to MongoDB############
    # i = 0
    # for x in horse_ID:
        #intialize the GPS for the STIL Drone
        # lon = the_connection.messages['GPS_RAW_INT'].lon
        # lat = the_connection.messages['GPS_RAW_INT'].lat
        # lons = str(lon)
        # lats = str(lat)
        #tracked horses data
        # post = {"horse_detection_code": x, "Longitudinal": (lons[:3]+'.'+lons[3:]), "Latitude": (lats[:3]+'.'+lats[3:]),"Date" : str(cdate), "Time" : str(ctime)}
        # collection.insert_one(post)
        # collection_horse_eating.insert_one(post)
        # collection_horse_sleeping.insert_one(post)
        # i += 1
    #########posting to MongoDB#########
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

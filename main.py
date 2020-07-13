"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network(args.model, args.device, args.prob_threshold)
    # Set Probability threshold for detections
    #prob_threshold = args.prob_threshold
    video_file=args.input
    output_path="results"
    single_img = False
    #model_name=args.model
    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.cpu_extension)
    ### TODO: Handle the input stream ###
    # Handle the input stream
    if video_file == 'CAM': # Check for live feed
        video_file = 0

    elif video_file.endswith('.jpg') or video_file.endswith('.bmp') :    # Check for input image
        single_img = True

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
    print("03")
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    #out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)

    counter=0
    num_people=0
    people_count=0
    total_people=0
    people_count_av =1
    count_av=1
    average_ten=[]
    duration_list=[]
    is_diff=False
    is_inf=False
    start_count_duration = 0
    stop_duration_count = 0
    timer_list=[]
    duration=0

    #start_inference_time=time.time()

    try:
        infer_network.set_initial(initial_w, initial_h)
        ### TODO: Loop until stream is over ###
        while cap.isOpened():
            ### TODO: Read from the video capture ###
            ret, frame=cap.read()
            if not ret:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            counter+=1

            ### TODO: Start asynchronous inference for specified request ###
            ### TODO: Get the results of the inference request ###
            start_infer_timer=time.time()

            coords, image = infer_network.predict(frame)

            stop_infer_timer=time.time()

            infer_time=stop_infer_timer-start_infer_timer
            timer_list.append(infer_time)

            ### TODO: Extract any desired stats from the results ###
                ### TODO: Calculate and send relevant information on ###
                ### current_count, total_count and duration to the MQTT server ###
                ### Topic "person": keys of "count" and "total" ###
                ### Topic "person/duration": key of "duration" ###


            ###########################################

            FOR REVIEWER : sorry submit a bit quickly!-(

            currently make modif - sorry






            ######################################


            num_people= len(coords) # number of people detected
            if num_people > people_count_av: # if new people
                # to take into account the first people detected we add a 0 duration
                # side effect due to the fact that the total number of people is counted for each new duration added (in ui)
                if total_people==0: # so, if first people detected
                    client.publish("person/duration", json.dumps({"duration": 0}))# the duration 0 is added to have the first people added in total people (in ui)
                total_people += num_people-people_count
                start_count_duration = counter

            if  is_diff and is_inf: # if number people is different and inferior to last 10 frames (average)
                stop_duration_count = counter
                duration=(stop_duration_count - start_count_duration)/fps # count(=frame number)/fps is used to get real time (not computational time)
                client.publish("person/duration", json.dumps({"duration": duration}))# duration is added and total_people incremented (in ui)
                duration_list.append(stop_duration_count - start_count_duration)
                start_count_duration = 0
                stop_duration_count = 0

            average_ten.append(num_people) #average of 10 frames is used to avoid some false detection
            if counter %10 == 0:"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network(args.model, args.device, args.prob_threshold)
    # Set Probability threshold for detections
    #prob_threshold = args.prob_threshold
    video_file=args.input
    output_path="results"
    single_img = False
    #model_name=args.model
    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.cpu_extension)
    ### TODO: Handle the input stream ###
    # Handle the input stream
    if video_file == 'CAM': # Check for live feed
        video_file = 0

    elif video_file.endswith('.jpg') or video_file.endswith('.bmp') :    # Check for input image
        single_img = True

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
    print("03")
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    #out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)

    counter=0 # num frame
    num_people=0 # number people detected
    people_count=0
    total_people=0
    people_count_av =0 # average on 10 last frame
    count_av=0
    average_ten=[] # list to make average num people
    duration_list=[]
    is_diff=False
    is_inf=False
    start_count_duration = 0
    stop_duration_count = 0
    timer_list=[]
    duration=0

    #start_inference_time=time.time()

    try:
        infer_network.set_initial(initial_w, initial_h)
        ### TODO: Loop until stream is over ###
        while cap.isOpened():
            ### TODO: Read from the video capture ###
            ret, frame=cap.read()
            if not ret:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            counter+=1

            ### TODO: Start asynchronous inference for specified request ###
            ### TODO: Get the results of the inference request ###
            start_infer_timer=time.time()

            coords, image = infer_network.predict(frame)

            stop_infer_timer=time.time()

            infer_time=stop_infer_timer-start_infer_timer
            timer_list.append(infer_time)

            ### TODO: Extract any desired stats from the results ###
                ### TODO: Calculate and send relevant information on ###
                ### current_count, total_count and duration to the MQTT server ###
                ### Topic "person": keys of "count" and "total" ###
                ### Topic "person/duration": key of "duration" ###

            num_people= len(coords) # number of people detected
            if num_people > people_count_av: # if new people (compared to last 10 frames)
                total_people += num_people-people_count
                start_count_duration = counter

            if  is_diff and is_inf: # if number people is different and inferior to last 10 frames (average)
                stop_duration_count = counter
                duration=(stop_duration_count - start_count_duration)/fps # count(=frame number)/fps is used to get real time (not computational time)
                client.publish("person/duration", json.dumps({"duration": duration}))# duration is added and total_people incremented (in ui)
                duration_list.append(stop_duration_count - start_count_duration)
                start_count_duration = 0
                stop_duration_count = 0

            average_ten.append(num_people) #average of 10 frames is used to avoid some false detection
            if counter %10 == 0:
                if sum(average_ten)==0:
                    people_count_av =0
                else:
                    people_count_av = round(sum(average_ten)/len(average_ten))
                average_ten[:] = []
            is_diff= (people_count_av != count_av)
            is_inf= (people_count_av < count_av)
            count_av = people_count_av
            people_count=num_people

            ##################
            client.publish("person", json.dumps({"count": num_people}))
            ##################tempo
            #print(f"Number of People in frame = {len(coords)}")
            #print(f"Total number of people = {total_people}")
            #print(f"Inference time = {infer_time}")

            #out_video.write(image)
            #cv2.imshow("video", image)
            ##################

            ### TODO: Send the frame to the FFMPEG server ###
            sys.stdout.buffer.write(frame)
            sys.stdout.flush()
        ### TODO: Write an output image if `single_image_mode` ###
        if single_image:
                cv2.imwrite('out.png', frame)
        ###########
        #Average_infer_time=(sum(timer_list)/len(timer_list))
        #total_Average_duration=((sum(duration_list)/len(duration_list))/fps)
        #print(total_Average_duration)
        #print("average time : ",Average_infer_time*1000," ms")
        #print("total: ",total_people)

    except Exception as e:
        ''#print("Could not run Inference: ", e)

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)
    #infer_on_stream(args, None)


if __name__ == '__main__':
    main()
                if sum(average_ten)==0:
                    people_count_av =0
                else:
                    people_count_av = round(sum(average_ten)/len(average_ten))
                average_ten[:] = []
            is_diff= (people_count_av != count_av)
            is_inf= (people_count_av < count_av)
            count_av = people_count_av
            people_count=num_people

            ##################
            client.publish("person", json.dumps({"count": num_people}))
            ##################tempo
            #print(f"Number of People in frame = {len(coords)}")
            #print(f"Total number of people = {total_people}")
            #print(f"Inference time = {infer_time}")

            #out_video.write(image)
            #cv2.imshow("video", image)
            ##################

            ### TODO: Send the frame to the FFMPEG server ###
            sys.stdout.buffer.write(frame)
            sys.stdout.flush()
        ### TODO: Write an output image if `single_image_mode` ###
        if single_image:
                cv2.imwrite('out.png', frame)
        ###########
        #Average_infer_time=(sum(timer_list)/len(timer_list))
        #total_Average_duration=((sum(duration_list)/len(duration_list))/fps)
        #print(total_Average_duration)
        #print("average time : ",Average_infer_time*1000," ms")
        #print("total: ",total_people)

    except Exception as e:
        ''#print("Could not run Inference: ", e)

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)
    #infer_on_stream(args, None)


if __name__ == '__main__':
    main()

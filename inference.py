#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore
import time
import cv2


class Network:
    """
    Load and configure inference plugins for the specified target devices
    and performs synchronous and asynchronous modes for the specified infer requests.
    """
    #log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    
    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.5):
        ### TODO: Initialize any class variables desired ###
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold
        self.extensions=extensions

        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self, cpu_extension=None):
        ### TODO: Load the model ###
        '''
        Load the model
        '''
        # Load IE API
        #log.info("Creating Inference Engine...")
        self.ie = IECore()

        if cpu_extension and 'CPU' in self.device:
            self.ie.add_extension(cpu_extension, "CPU")          
        
        # Read IR
        #log.info("Loading network files:\n\t{}\n\t{}".format(self.model_structure, self.model_weights))
        self.net = IENetwork(model=self.model_structure, weights=self.model_weights)
        #self.net = self.ie.read_network(model=self.model_structure, weights=self.model_weights)
        
        ### TODO: Check for supported layers ###
        if "CPU" in self.device:
            supported_layers = self.ie.query_network(self.net, "CPU")
            not_supported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("Layers are not supported {}:\n {}".
                      format(self.device, ', '.join(not_supported_layers)))
                log.error("Specify cpu extensions using -l")
                sys.exit(1)      
                
        # Load IR to the plugin
        #log.info("Loading IR to the plugin...")
        self.exec_net = self.ie.load_network(network=self.net, num_requests=1, device_name=self.device)

        self.input_blob=next(iter(self.exec_net.inputs))
        self.output_blob=next(iter(self.exec_net.outputs))

        ### TODO: Return the loaded inference plugin ###
        #return self.exec_net

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        n, c, h, w = self.input_shape
        return (n, c, h, w)

    def predict(self, image):
        ### TODO: Start an asynchronous request ###
        '''
        Perform inference
        '''
        
        #log.info("Performing inference...")
        feed_dict = self.preprocess_input(image)
        outputs=self.exec_net.start_async(request_id=0, inputs=feed_dict)
        while True:
            status=self.exec_net.requests[0].wait(-1)
            if status==0:
                break
            else: time.sleep(1)
        coords=self.get_output(outputs)
        #self.draw_outputs(coords, image)
        ### TODO: Return any necessary information ###
        return coords, image

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        while True:
            status=self.exec_net.requests[0].wait(-1)
            if status==0:
                break
            else: time.sleep(1)

    def get_output(self, outputs):
        ### TODO: Extract and return the output results
        '''
        Get Bounding Boxs coordinates
        '''
        res = self.exec_net.requests[0].outputs[self.output_blob]
        coords=[]
        for obj in res[0][0]:
            if obj[2] > self.threshold:
                xmin = int(obj[3] * self.initial_w)
                ymin = int(obj[4] * self.initial_h)
                xmax = int(obj[5] * self.initial_w)
                ymax = int(obj[6] * self.initial_h)
                coords.append((xmin, ymin, xmax, ymax))
        return coords

    def preprocess_input(self, image):
        '''
        Preprocess the input images
        '''
        #log.info("Preprocessing the input images...")
        input_dict={}
        n, c, h, w = self.input_shape
        in_frame = cv2.resize(image, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((n, c, h, w))
        input_dict[self.input_name] = in_frame
        return input_dict
        raise NotImplementedError

    def set_initial(self, w, h):
        self.initial_w = w
        self.initial_h = h
        
    def draw_outputs(self, coords, image):
        '''
        Draw Bounding Boxs and texts on images
        '''
        #xmin, ymin, xmax, ymax= self.preprocess_outputs(outputs)
        color = (10, 245, 10)
        #cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        for obj in coords:
            cv2.rectangle(image, (obj[0], obj[1]), (obj[2], obj[3]), color, 2)
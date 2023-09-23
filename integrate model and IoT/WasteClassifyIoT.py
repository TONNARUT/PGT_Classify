import cv2
import os, time
from datetime import datetime

from keras import models
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

import numpy as np
import matplotlib.pyplot as plt
import pickle

import microgear.client as client

def saveimgfile(frame):
    #In Windows, ':' is invalid character for a file name. 
    filename = 'img' + datetime.now().strftime("%Y_%m_%d-%I-%M-%S_%p")
    print('filename',filename)
    if not cv2.imwrite(saveimg_path + "/" + filename + ".png",frame):
        raise Exception("Could not write image")

def wastepredict_h5(frame):
    # ===============================================
    # * * * Predict image class from captured image * * *
    # 0: AluCan, 1: Glass, 2: Plastic
    # ===============================================     
    #Convert the captured frame (BGR) into RGB (convert a NumPy array to an image)        
    im = Image.fromarray(frame, 'RGB')
    
    #Resizing into 128x128 because we trained the model with this image size.        
    im = im.resize((128,128))
    
    #Changing dimension 128x128x3 
    img_array = np.array(im) 
    
    #Our keras model used a 4D tensor, (images x height x width x channel)
    #So changing dimension 128x128x3 into 1x128x128x3
    img_array = np.expand_dims(img_array, axis=0)
    
    #Prepare input array to predict
    img_array = preprocess_input(img_array)

    # OR we can use the following code section
    pred_result = my_model.predict(img_array)
    waste_type = class_names[np.argmax(pred_result, -1)[0]]
    print(waste_type)
    return(waste_type)  

def wastepredict_tflite(frame):
    # ===============================================
    # * * * Predict image class from captured image * * *
    # Use TensorFlow Lite to Predict
    # 0: AluCan, 1: Glass, 2: Plastic
    # ===============================================     
    #Convert the captured frame (BGR) into RGB (convert a NumPy array to an image)        
    im = Image.fromarray(frame, 'RGB')
    
    #Resizing into 128x128 because we trained the model with this image size.        
    im = im.resize((128,128))
    
    #Changing dimension 128x128x3 
    img_array = np.array(im) 
    
    #Our keras model used a 4D tensor, (images x height x width x channel)
    #So changing dimension 128x128x3 into 1x128x128x3
    img_array = np.expand_dims(img_array, axis=0)
    
    #Prepare input array to predict
    img_array = preprocess_input(img_array)

    #Calling the predict method on model to predict on the image   
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input = interpreter.get_input_details()
    input_shape = input[0]['shape']

    input_tensor_index = input[0]["index"]
    output = interpreter.tensor(interpreter.get_output_details()[0]["index"])

    interpreter.set_tensor(input_tensor_index, img_array) 

    time_start = time.time()
    interpreter.invoke()

    time_end = time.time()
    total_tflite_time = time_end - time_start
    print("Total prediction time: ", total_tflite_time)

    digit = np.argmax(output()[0])
    #print(digit)
    waste_type = class_names[digit]
    print(waste_type)
  
    return(waste_type)


def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("@msg/data")

def on_message(client, userdata, msg):    
    print(msg.topic+" "+str(msg.payload))
    client_id = "a4ae1efd-41db-4773-9e14-2827bab6abf2"
    token = "2U7u7LyCj5DC9Vub4oD627LaYSo3SsTt"
    secret = "Aen$azBlLA4CZ$cgmapYy)Vb)JdbQSEJ"
    broker = "mqtt.netpie.io"
    port = 1883
    client = mqtt.Client(client_id)
    client.username_pw_set(token, secret)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker,port)

def garbage_publishing(prediction):
    if prediction == "Plastic":
        client.publish("@msg/data","Plastic")
    elif prediction == "Glass":
        client.publish("@msg/data","Glass")
    else:
        client.publish("@msg/data","AluCan")  

if __name__ == '__main__':
    # Check current working directory.
    # retval = os.getcwd()
    # print("Current working directory: ",retval)

    # Set output directory to save image snapshot if it deos not exist
    saveimg_path = 'd:/saveimage'
    if not os.path.exists(saveimg_path):
        os.makedirs(saveimg_path)

    # Load class configuration (0:Alucan, 1:Glass, 2:Plastic)
    file_name = "d:/IoT/classname.pkl"
    open_file = open(file_name, "rb")
    class_names = pickle.load(open_file)
    open_file.close()
    class_names  
    #print(class_names)

    # Load pgt h5 model (to predict)
    my_model = load_model('d:/IoT/pgt_model.h5') # load model h5

    # Load pgt tflite model (to predict)
    tflite_path =  'd:/IoT/pgt_model.tflite' # load model tflite   

    # Choose loaded model
    model_choose = int(input("Please select (1) .h5 model or (2) .tflite model or (3) to quit: "))
    while model_choose < 1 and model_choose > 3:
        model_choose = input("Please select (1) .h5 model or (2) .tflite model or (3) to quit: ")
    if model_choose == 3:
        sys.exit()

    # Save a picture snapshot manually or automaticall every defined period of time
    sec = 0

    ans = input ("Do you also want to save a picture? (y/n) ")
    if ans.upper() == "Y":
        sec = int(input("Save a picture at every ... seconds [0: every second - elapsed time] "))     

    # Open the device at the ID 0
    #cap = cv2.VideoCapture(0) #Camera Channel 0 
    cap = cv2.VideoCapture(1) #Camera channel 1

    # Check whether user selected camera is opened successfully.
    if not (cap.isOpened()):
        print("Could not open video device")

    cap.set(3,640/2) #width=640/2
    cap.set(4,480/2) #height=480/2
    
    # Set up automatic timer to wait in seconds between each object prediction 
    wait_sec = 5
    t1_time = datetime.now()
    timeout_wait = True

    while(True):
        start_time = time.time() # for saved image file name purpose     
        # Capture frame-by-frame: turn video frame into numpy ndarray
        ret, frame = cap.read()
        # Display the resulting frame
        cv2.imshow('preview',frame)

        # Set up the waitKey (keyboard) for checking multiple times   
        keyboard = cv2.waitKey(1) & 0xFF 

        if ans.upper() == "Y": # save a picture manually
            if sec == 0: # save a picture automatically by default
                end_time = time.time() 
                time.sleep(1.0 - (end_time - start_time))
                saveimgfile(frame)
                #time.sleep(1.0 - time.time() + start_time) # Sleep for 1 second minus elapsed time
            else: # save a picture automatically at every x seconds
                time.sleep(sec - time.time() + start_time) # Sleep for x second minus elapsed time        
                saveimgfile(frame)
        # Waits for a user input to save the image           
        elif keyboard == ord('y'):
                saveimgfile(frame)     

        # Waits for a user input to quit the application
        if keyboard == ord('q'):
            break    
        if keyboard == ord('p'): #pause program
           cont = input("Pause, do you want to continue (Y/N)? ")
           if not (cont.upper() == "Y"):
              break
        elif keyboard == 32:  # if space bar, then predicting object immediately without delay
            timeout_wait = True

        # Next improvement: add sensor to detect object existed, 
        #                   and execute the prediction only if the object exists (timeout_wait = True).         

        if timeout_wait: # space bar or wait for 4 seconds

            # choose between h5 and tflite model 
            if model_choose == 1:
                waste_category = wastepredict_h5(frame)        #large size predict model
            else:
                waste_category = wastepredict_tflite(frame)    #small size predict model

            # waste_category can be used for further action (e.g., IoT application, robotic programming, etc.)
            garbage_publishing(waste_category)           
            t1_time = datetime.now()
            timeout_wait = False
        
        # Calculate the wait time (seconds) after previously predicting the object  
        t2_time = datetime.now()
        #print((t2_time - t1_time).total_seconds())
        if ((t2_time - t1_time).total_seconds() >= wait_sec):
            t1_time = t2_time
            timeout_wait = True       
            #print("wait %d seconds done " % wait_sec)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
'''
Use case for this conversion script is described as below:
1) model = 'mobilenet', stride = 8, 16, tensor_size = 1 x 3 x stride x stride
'''

import tensorflow as tf
import math
import tflite2onnx
import numpy as np
import onnxruntime
import argparse
import os
from posenet_factory import load_model

tf.compat.v1.disable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='resnet50')  # mobilenet resnet50
parser.add_argument('--stride', type=int, default=16)  # 8, 16, 32 (max 16 for mobilenet)
parser.add_argument('--tensor_size', type=int, default=1* 3 * 257 *257)
parser.add_argument('--hint', type=bool, default=False)
args = parser.parse_args()
output_stride = args.stride
model_name = args.model # mobilenet or resnet. But resnet not supported yet
input_tensor_size = args.tensor_size

def test_onnx(input_model_file, input_tensor_size):
    session = onnxruntime.InferenceSession(input_model_file, None)
    input_width = input_heigh = check_input_tensor_size(input_tensor_size, output_stride)
    input_data = np.array(np.random.random_sample(input_tensor_size), dtype=np.float32)
    input_data =input_data.reshape(1, 3, input_heigh, input_width)
    # get the name of the first input of the model
    input_name = session.get_inputs()[0].name
    
    raw_result = session.run([], {input_name: input_data})
    print(raw_result)
    print ("ONNX model file is valid\n")
def convert_pb2tflite(model_name, savedmodel_path, input_tensor_size, output_stride):
    model = tf.saved_model.load(savedmodel_path)
    concrete_func = model.signatures[
    tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    input_width = input_heigh = check_input_tensor_size(input_tensor_size, output_stride)
    if ((input_width < 0) | (input_width < 0)):
        print("Convert pb to tflite fail\n")
    
    concrete_func.inputs[0].set_shape([1, input_width, input_heigh, 3])
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    tflite_model = converter.convert()
    export_model_file = savedmodel_path + '/' + model_name + '_stride' + str(output_stride) \
    + '_imagesize' + str(input_width) + '.tflite'
    with tf.io.gfile.GFile(export_model_file, 'wb') as f:
        f.write(tflite_model)
    test_tflite(export_model_file)
    print("Convert savemodel to TFLite successfully\n")
    return export_model_file
def convert_tflite2onnx(input_model_file, input_tensor_size):
    onnx_path = input_model_file[:-6] + 'onnx'
    tflite2onnx.convert(input_model_file, onnx_path)
    #test
    test_onnx(onnx_path, input_tensor_size)
    print("Convert TFLite to ONNX successfully\n")
    return onnx_path
def convert_pb2onnx(model_name, savedmodel_path, output_stride):
    export_model_file = savedmodel_path + '/' + model_name + '_stride' + str(output_stride) \
    + '.onnx'
    cmd_line = 'python -m tf2onnx.convert --saved-model '
    cmd_line +=  savedmodel_path
    #cmd_line += ' --inputs-as-nchw  sub_2:0 --output '
    cmd_line += '  --output '
    cmd_line += export_model_file
    print('cmd_line is ', cmd_line)
    os.system(cmd_line)
    return export_model_file

def test_tflite(input_model_file):
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=input_model_file)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    #print(output_data)
    #print("TFlite model is valid\n")
    
    #check  some api called in onnx convert
def load_savemodel(model, stride):
    quant_bytes = 4
    multiplier = 1
    return load_model(model, stride, quant_bytes, multiplier)
    
def check_input_tensor_size(input_tensor_size, output_stride):
    input_width = input_heigh = math.sqrt(input_tensor_size/3)
    #check valid input weight and heigh 
    if (((input_width - 1)/output_stride) * output_stride !=  (input_width - 1)) \
    | (((input_heigh - 1)/output_stride) * output_stride !=  (input_heigh - 1)):
        print ("model input size not supported\n")
        print ("It should be: output stride x n (integer) \n")
        return -1
    return int(input_width)   
def main(): 
    if args.hint == True:
        print("Use case for this conversion script is described as below:")
        print("========================================================================")
        print("Usage for mobilenet:")
        print("===================")
        print("model = 'mobilenet'; stride = 8 or 16; tensor_size = 1 x 3 x XYZ x XYZ")
        print("with condition is stride value divides XYZ")
        print("- Command is used")
        print("Ex: python3 convert.py --model mobilenet --stride 16 --tensor_size 49923")
        print("Ex: python3 convert.py --model mobilenet --stride 16 --tensor_size 198147")
        print("Ex: python3 convert.py --model mobilenet --stride 16 --tensor_size 789507")
        print("========================================================================")
        print("Usage for resnet:")
        print("=================")
        print("model = 'resnet50'; stride = 32")
        print("- Command is used")
        print("python convert.py --model resnet50 --stride 32")
        print("========================================================================")
        return 0



    if (check_input_tensor_size(input_tensor_size, output_stride) < 0):
        return -1
    if (model_name == "resnet50") & (output_stride != 32):
        print("Only support resnet50 with output stride 32")
        return -1
    #Load saved model from tfjs
    savedmodel_path = load_model(model_name, output_stride)    

    if (model_name == "mobilenet"):


        #Convert saved model to TFLite
        tflite_model_file = convert_pb2tflite(model_name, savedmodel_path, input_tensor_size, output_stride)

        #Convert TFLite to ONNX
        onnx_model_file = convert_tflite2onnx(tflite_model_file, input_tensor_size)

    elif (model_name == "resnet50"):
        print("Warning: resnet50 is converted with dynamic shape")
        print("input tensor size does not affect")
        onnx_model_file = convert_pb2onnx(model_name, savedmodel_path, output_stride)
    else:
        print("Model architecture only supports mobilenet, resnet50")
    print("SUMMARY")
    print("============================")
    print("savedmodel converted path  :", savedmodel_path)
    #if (model_name == "mobilenet"):
    #    print("TFLite model converted path:", tflite_model_file)
    print("ONNX model converted path  :", onnx_model_file)

    print("test ONNX model after converting")
    #test_onnx(onnx_model_file, 198147)
    test_onnx(onnx_model_file, 789507)

if __name__ == "__main__":
    main()

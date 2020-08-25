import tensorflow as tf
import os
import converter.config as config
import converter.tfjs2tf as tfjs2tf

def load_model(model, stride, quant_bytes=4, multiplier=1.0):

    if model == config.RESNET50_MODEL:
        model_cfg = config.bodypix_resnet50_config(stride, quant_bytes)
        print('Loading ResNet50 model')
    else:
        model_cfg = config.bodypix_mobilenet_config(stride, quant_bytes, multiplier)
        print('Loading MobileNet model')

    model_path = model_cfg['tf_dir']
    if not os.path.exists(model_path):
        print('Cannot find tf model path %s, converting from tfjs...' % model_path)
        tfjs2tf.convert(model_cfg)
        assert os.path.exists(model_path)

    loaded_model = tf.saved_model.load(model_path)

    signature_key = tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    print('We use the signature key %s It should be in the keys list:' % signature_key)
    for sig in loaded_model.signatures.keys():
        print('signature key: %s' % sig)

    model_function = loaded_model.signatures[signature_key]
    print('model outputs: %s' % model_function.structured_outputs)

    return model_path

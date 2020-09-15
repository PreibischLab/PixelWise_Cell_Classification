from tensorflow.keras import backend as K

def diceCoef(y_true, y_pred, smooth=1):
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.flatten(y_true)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth)/(K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def IoULoss(y_true, y_pred):
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.flatten(y_true)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return 1-(intersection / union)


# class MaskMeanIoU(MeanIoU):
# #   """Mean Intersection over Union """
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         print('hello')
#         y_pred = tf.math.argmax(y_pred, axis=-1)
#         print('y_true= {} y_pre={}'.format(y_true.shape, y_pred.shape))
#         return super().update_state(y_true, y_pred, sample_weight=sample_weight)

def dice_loss(onehots_true, logits):
    probabilities = tf.nn.softmax(logits)
    #weights = 1.0 / ((tf.reduce_sum(onehots_true, axis=0)**2) + 1e-3)
    #weights = tf.clip_by_value(weights, 1e-17, 1.0 - 1e-7)
    numerator = tf.reduce_sum(onehots_true * probabilities, axis=0)
    #numerator = tf.reduce_sum(weights * numerator)
    denominator = tf.reduce_sum(onehots_true + probabilities, axis=0)
    #denominator = tf.reduce_sum(weights * denominator)
    loss = 1.0 - (2.0 * numerator + 1) / (denominator + 1)
    return loss

def IoULoss(targets, inputs, smooth=1e-6):
    
    #flatten label and prediction tensors
#     inputs = tf.nn.softmax(inputs)
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
#     intersection = tf.reduce_sum(targets * inputs, axis=0)
    intersection = K.sum(targets * inputs)
#     intersection = K.sum(K.dot(targets, inputs))
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection
    
    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU
# def IoULoss(targets, inputs, smooth=1e-6):
    
#     #flatten label and prediction tensors
#     inputs = K.flatten(inputs)
#     targets = K.flatten(targets)
    
#     intersection = K.sum(K.dot(targets, inputs))
#     total = K.sum(targets) + K.sum(inputs)
#     union = total - intersection
    
#     IoU = (intersection + smooth) / (union + smooth)
#     return 1 - IoU

# class MaskMeanIoU(MeanIoU):
# #   """Mean Intersection over Union """
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         print('hello')
#         y_pred = tf.math.argmax(y_pred, axis=-1)
#         print('y_true= {} y_pre={}'.format(y_true.shape, y_pred.shape))
#         return super().update_state(y_true, y_pred, sample_weight=sample_weight)
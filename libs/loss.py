import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16
import torch
import torchvision.models as models


def get_vgg16_weights(save_dir):
  pytorch_model = models.vgg16(pretrained=True)

  # select weights in the conv2d layers and transpose them to keras dim ordering:
  wblist_torch = list(pytorch_model.parameters())[:26]
  wblist_keras = []
  for i in range(len(wblist_torch)):
    if wblist_torch[i].dim() == 4:
      w = np.transpose(wblist_torch[i].detach().numpy(), axes=[2, 3, 1, 0])
      wblist_keras.append(w)
    elif wblist_torch[i].dim() == 1:
      b = wblist_torch[i].detach().numpy()
      wblist_keras.append(b)
    else:
      raise Exception('Fully connected layers are not implemented.')

  keras_model = VGG16(include_top=False, weights=None)
  keras_model.set_weights(wblist_keras)
  keras_model.save_weights(save_dir + '/vgg16_pytorch2keras.h5')


def get_vgg16_model(weights):
    """ Creates a vgg model that returns a list of intermediate output values."""
    vgg = tf.keras.applications.VGG16(include_top=False, weights=weights)
    vgg.trainable = False

    outputs = [
        vgg.get_layer(name).output
        for name in ['block1_pool', 'block2_pool', 'block3_pool']
    ]
    return Model(vgg.input, outputs)


class StyleModel(Model):
    """ Build a model that returns the style and content tensors."""
    def __init__(self, weights):  #, style_layers):
        super(StyleModel, self).__init__()
        #self.weights = weights
        self.vgg = get_vgg16_model(weights=weights)
        #self.style_layers = style_layers
        self.vgg.trainable = False
        # Scaling for VGG input
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def call(self, inputs):
        preprocessed_input = tf.identity(inputs)
        Lambda(lambda x: (x - self.mean) / self.std)(preprocessed_input)
        outputs = self.vgg(preprocessed_input)
        return outputs


def loss_l1(y_true, y_pred):
    """
    Size-averaged L1 loss used in all the losses.

    If size_average is True, the l1 losses are means,
    If size_average is False, the l1 losses are sums divided by norm (should be specified),
        only have effect if y_true.ndim = 4.
    """
    if K.ndim(y_true) == 4:
        # images and vgg features
        return K.mean(K.abs(y_pred - y_true), axis=[1, 2, 3])
    elif K.ndim(y_true) == 3:
        # gram matrices
        return K.mean(K.abs(y_pred - y_true), axis=[1, 2])
    else:
        raise NotImplementedError(
            "Calculating L1 loss on 1D tensors? should not occur for this network"
        )


def gram_matrix(x, norm_by_channels=False):
    """Calculate gram matrix used in style loss"""
    # Assertions on input
    assert K.ndim(x) == 4, 'Input tensor should be a 4d (B, H, W, C) tensor'
    assert K.image_data_format(
    ) == 'channels_last', "Please use channels-last format"

    # Permute channels and get resulting shape
    x = K.permute_dimensions(x, (0, 3, 1, 2))
    shape = K.shape(x)
    B, C, H, W = shape[0], shape[1], shape[2], shape[3]

    # Reshape x and do batch dot product
    features = K.reshape(x, K.stack([B, C, H * W]))
    gram = K.batch_dot(features, features, axes=2)

    # Normalize with channels, height and width
    gram = gram / K.cast(C * H * W, x.dtype)
    return gram


def loss_per_pixel(mask, y_true, y_pred):
    """Pixel L1 loss outside the hole / mask"""
    assert K.ndim(y_true) == 4, 'Input tensor should be 4D (B, H, W, C).'
    return K.mean(K.abs(mask * (y_pred - y_true)), axis=[1, 2, 3])


def loss_perceptual(vgg_out, vgg_gt, vgg_comp):
    """Perceptual loss based on VGG16, see. eq. 3 in paper"""
    l = 0
    for o, c, g in zip(vgg_out, vgg_comp, vgg_gt):
        l += loss_l1(o, g) + loss_l1(c, g)
    return l


def loss_style(vgg_out, vgg_gt, vgg_comp):
    """Style loss consisting of two terms: out and comp."""
    style_score = 0.
    for o, g, c in zip(vgg_out, vgg_gt, vgg_comp):
        #print("shapes", o.shape, c.shape, g.shape, "dim", K.ndim(o), K.ndim(c), K.ndim(g))
        gram_gt = gram_matrix(g)
        #print('gram_gt', gram_gt.shape, 'gram_out', gram_matrix(o).shape, gram_matrix(c).shape)
        style_score += loss_l1(gram_matrix(o), gram_gt) + loss_l1(
            gram_matrix(c), gram_gt)
    return style_score


def loss_tv(inv_mask, y_comp):
    """Total variation loss, used for smoothing the hole region, see. eq. 6"""
    assert K.ndim(y_comp) == 4 and K.ndim(
        inv_mask) == 4, 'Input tensors should be 4D (B, H, W, C).'
    ## Create dilated hole region using a 3x3 kernel of all 1s.
    kernel = tf.ones([3, 3, 3, 3], tf.float32)
    dilated_mask = K.conv2d(inv_mask,
                            kernel,
                            data_format='channels_last',
                            padding='same')

    ## Cast values to be [0., 1.], and compute dilated hole region of y_comp
    dilated_mask = K.cast(K.greater(dilated_mask, 0), 'float32')
    P = dilated_mask * y_comp
    ## Calculate total variation loss
    return loss_l1(P[:, 1:, :, :], P[:, :-1, :, :]) + loss_l1(
        P[:, :, 1:, :], P[:, :, :-1, :])


def loss_total(model, inputs, targets, vgg16, training=True):
    mask = inputs[1]
    y_ = model(inputs, training=training)

    def loss(y_true, y_pred):
        # The raw output image Iout, but with the non-hole pixels directly set to ground truth
        y_comp = mask * y_true + (1 - mask) * y_pred

        style_pred = vgg16(y_pred)
        style_gt = vgg16(y_true)
        style_comp = vgg16(y_comp)

        # Compute loss components
        l_valid = loss_per_pixel(mask, y_true, y_pred)
        l_hole = loss_per_pixel((1 - mask), y_true, y_pred)
        l_perc = loss_perceptual(style_pred, style_gt, style_comp)

        l_style = loss_style(style_pred, style_gt, style_comp)
        l_tv = loss_tv((1 - mask), y_comp)
        # Return loss function
        return l_valid + 6 * l_hole + 0.05 * l_perc + 120 * l_style + 0.1 * l_tv
    return loss(targets, y_)

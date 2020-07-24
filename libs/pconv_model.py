from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Lambda, BatchNormalization, Activation, UpSampling2D, Concatenate, ReLU, LeakyReLU, Conv2D

from pconv2d_layer import PConv2D


def build_pconv_unet(img_shape, fine_tuning=False, train_bn=True):
    # INPUTS
    inputs_img = Input((img_shape, img_shape, 3), name='inputs_img')
    inputs_mask = Input((img_shape, img_shape, 3), name='inputs_mask')

    # ENCODER
    def encoder_block(img_in,
                      mask_in,
                      filters,
                      kernel_size,
                      batch_norm=True,
                      freeze_bn=False,
                      count=''):
        if count != '':
            count = '_' + count

        pconv, mask = PConv2D(filters,
                              kernel_size,
                              strides=2,
                              padding='same',
                              name='pconv2d_enc' + count)([img_in, mask_in])
        if batch_norm:
            pconv = BatchNormalization(name='bn_enc' + count)(
                pconv, training=not freeze_bn)
        pconv = Activation('relu')(pconv)

        return pconv, mask

    e_conv1, e_mask1 = encoder_block(inputs_img,
                                     inputs_mask,
                                     64,
                                     7,
                                     batch_norm=False,
                                     count='1')
    e_conv2, e_mask2 = encoder_block(e_conv1,
                                     e_mask1,
                                     128,
                                     5,
                                     freeze_bn=fine_tuning,
                                     count='2')
    e_conv3, e_mask3 = encoder_block(e_conv2,
                                     e_mask2,
                                     256,
                                     5,
                                     freeze_bn=fine_tuning,
                                     count='3')
    e_conv4, e_mask4 = encoder_block(e_conv3,
                                     e_mask3,
                                     512,
                                     3,
                                     freeze_bn=fine_tuning,
                                     count='4')
    e_conv5, e_mask5 = encoder_block(e_conv4,
                                     e_conv4,
                                     512,
                                     3,
                                     freeze_bn=fine_tuning,
                                     count='5')
    e_conv6, e_mask6 = encoder_block(e_conv5,
                                     e_mask5,
                                     512,
                                     3,
                                     freeze_bn=fine_tuning,
                                     count='6')
    e_conv7, e_mask7 = encoder_block(e_conv6,
                                     e_mask6,
                                     512,
                                     3,
                                     freeze_bn=fine_tuning,
                                     count='7')
    e_conv8, e_mask8 = encoder_block(e_conv7,
                                     e_mask7,
                                     512,
                                     3,
                                     freeze_bn=fine_tuning,
                                     count='8')

    # DECODER
    def decoder_block(img_in,
                      mask_in,
                      e_conv,
                      e_mask,
                      filters,
                      kernel_size,
                      batch_norm=True,
                      count=''):
        if count != '':
            count = '_' + count

        up_img = UpSampling2D(size=(2, 2),
                              name='img_upsamp_dec' + count)(img_in)
        up_mask = UpSampling2D(size=(2, 2),
                               name='mask_upsamp_dec' + count)(mask_in)
        concat_img = Concatenate(axis=3, name='img_concat_dec' +
                                 count)([e_conv, up_img])
        concat_mask = Concatenate(axis=3, name='mask_concat_dec' +
                                  count)([e_mask, up_mask])
        pconv, mask = PConv2D(filters,
                              kernel_size,
                              padding='same',
                              name='pconv2d_dec' +
                              count)([concat_img, concat_mask])

        if batch_norm:
            pconv = BatchNormalization(name='bn_dec' + count)(pconv)
        pconv = LeakyReLU(alpha=0.2, name='leaky_dec' + count)(pconv)
        return pconv, mask

    # DECODER
    d_conv9, d_mask9 = decoder_block(e_conv8,
                                     e_mask8,
                                     e_conv7,
                                     e_mask7,
                                     512,
                                     3,
                                     count='9')
    d_conv10, d_mask10 = decoder_block(d_conv9,
                                       d_mask9,
                                       e_conv6,
                                       e_mask6,
                                       512,
                                       3,
                                       count='10')
    d_conv11, d_mask11 = decoder_block(d_conv10,
                                       d_mask10,
                                       e_conv5,
                                       e_mask5,
                                       512,
                                       3,
                                       count='11')
    d_conv12, d_mask12 = decoder_block(d_conv11,
                                       d_mask11,
                                       e_mask4,
                                       e_mask4,
                                       512,
                                       3,
                                       count='12')
    d_conv13, d_mask13 = decoder_block(d_conv12,
                                       d_mask12,
                                       e_conv3,
                                       e_mask3,
                                       256,
                                       3,
                                       count='13')
    d_conv14, d_mask14 = decoder_block(d_conv13,
                                       d_mask13,
                                       e_conv2,
                                       e_mask2,
                                       128,
                                       3,
                                       count='14')
    d_conv15, d_mask15 = decoder_block(d_conv14,
                                       d_mask14,
                                       e_conv1,
                                       e_mask1,
                                       64,
                                       3,
                                       count='15')
    d_conv16, d_mask16 = decoder_block(d_conv15,
                                       d_mask15,
                                       inputs_img,
                                       inputs_mask,
                                       3,
                                       3,
                                       batch_norm=False,
                                       count='16')
    outputs = Conv2D(3, 1, activation='sigmoid', name='outputs_img')(d_conv16)

    # Setup the model inputs / outputs
    model = Model(inputs=[inputs_img, inputs_mask], outputs=outputs)

    # This will also freeze bn parameters `beta` and `gamma`:
    if fine_tuning:
        for l in model.layers:
            if 'bn_enc' in l.name:
                l.trainable = False

    return model

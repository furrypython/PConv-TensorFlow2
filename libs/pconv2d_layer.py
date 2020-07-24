from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, InputSpec


class PConv2D(Conv2D):
    def __init__(self, *args, **kwargs):
        super(PConv2D, self).__init__(*args, **kwargs)
        self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4)]

    def build(self, input_shape):
        """Adapted from original _Conv() layer of Keras
        param input_shape: list of dimensions for [img, mask]
        """
        assert self.data_format == 'channels_last', "data format should be `channels_last`"
        #self.input_dim = input_shape[0][-1]
        input_channel = self._get_input_channel(input_shape)
        # Image kernel
        kernel_shape = self.kernel_size + (input_channel, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='img_kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters, ),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        # Mask kernel
        self.kernel_mask = K.ones(shape=self.kernel_size +
                                  (input_channel, self.filters))

        # Calculate padding size to achieve zero-padding
        self.pconv_padding = (
            (int((self.kernel_size[0] - 1) / 2),
             int((self.kernel_size[0] - 1) / 2)),
            (int((self.kernel_size[0] - 1) / 2),
             int((self.kernel_size[0] - 1) / 2)),
        )

        # Window size - used for normalization
        self.window_size = self.kernel_size[0] * self.kernel_size[1]

        self.built = True

    def call(self, inputs):
        # Both image and mask must be supplied
        assert isinstance(inputs, list) and len(inputs) == 2

        # Padding done explicitly so that padding becomes part of the masked partial convolution
        images = K.spatial_2d_padding(inputs[0], self.pconv_padding,
                                      self.data_format)
        masks = K.spatial_2d_padding(inputs[1], self.pconv_padding,
                                     self.data_format)

        # Apply convolutions to image
        img_output = K.conv2d((images * masks),
                              self.kernel,
                              strides=self.strides,
                              padding='valid',
                              data_format=self.data_format,
                              dilation_rate=self.dilation_rate)

        # Apply convolutions to mask
        mask_output = K.conv2d(masks,
                               self.kernel_mask,
                               strides=self.strides,
                               padding='valid',
                               data_format=self.data_format,
                               dilation_rate=self.dilation_rate)

        # Calculate the mask ratio on each pixel in the output mask
        mask_ratio = self.window_size / (mask_output + 1e-8)
        # Clip output to be between 0 and 1
        mask_output = K.clip(mask_output, 0, 1)
        # Remove ratio values where there are holes
        mask_ratio = mask_ratio * mask_output
        # Normalize iamge output
        img_output = img_output * mask_ratio

        # Apply bias only to the image (if chosen to do so)
        if self.use_bias:
            img_output = K.bias_add(img_output,
                                    self.bias,
                                    data_format=self.data_format)

        # Apply activations on the image
        if self.activation is not None:
            img_output = self.activation(img_output)

        return [img_output, mask_output]

    def compute_output_shape(self, input_shape):
        space = input_shape[0][1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding='same',
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        new_shape = (input_shape[0][0], ) + tuple(new_space) + (self.filters, )
        return [new_shape, new_shape]

    def _get_input_channel(self, input_shape):
        channel_axis = -1
        if input_shape[0].dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        return int(input_shape[0][-1])

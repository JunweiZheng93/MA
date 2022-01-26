import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def get_model(category='chair', max_num_parts=4, attention=False, multi_inputs=False):

    # Input layer
    inputs = tf.keras.Input(shape=(32, 32, 32, 1), name='Input')

    # Decomposer
    binary_shape_encoder_outputs = BinaryShapeEncoder(name='BinaryShapeEncoder')(inputs)
    proj_outputs = [Projection(name=f'Projection{i}')(binary_shape_encoder_outputs) for i in range(max_num_parts)]

    if attention:
        if multi_inputs:
            # Composer
            part_decoder = PartDecoder(name='PartDecoder')
            part_decoder_outputs = [part_decoder(x) for x in proj_outputs]
            # stacked_decoded_parts.shape == (B, num_parts, H, W, D, C)
            stacked_decoded_parts = tf.stack(part_decoder_outputs, axis=1)
            # reshape_decoded_parts.shape == (B, num_parts, H*W*D*C)
            reshaped_decoded_parts = tf.reshape(stacked_decoded_parts, (tf.shape(stacked_decoded_parts)[0], max_num_parts, 32*32*32*1))
            # attention_outputs0.shape == (B, num_parts, 256)
            attention_outputs0 = AttentionLayer(num_blocks=1, num_heads=8, d_model=32*32*32, seq_len=max_num_parts, name='AttentionLayer0')(reshaped_decoded_parts)
            # proj_attention_inputs.shape == (B, num_parts, encoding_dimensions)
            proj_attention_inputs = tf.transpose(proj_outputs, (1, 0, 2))
            # attention_outputs1.shape == (B, num_parts, 256)
            attention_outputs1 = AttentionLayer(num_blocks=10, num_heads=8, d_model=100, seq_len=max_num_parts, name='AttentionLayer1')(proj_attention_inputs)
            # concat_outputs.shape == (B, num_parts, 512)
            concat_outputs = tf.concat([attention_outputs0, attention_outputs1], axis=2)
            # dense_outputs.shape == (B, num_parts, 12)
            dense_outputs = keras.layers.Dense(12, name='Dense')(concat_outputs)
            # theta.shape == (B, num_parts, 3, 4)
            theta = tf.reshape(dense_outputs, (tf.shape(dense_outputs)[0], max_num_parts, 3, 4))
            # resampling_outputs.shape == (B, num_parts, H, W, D, C)
            resampling_outputs = Resampling(name='Resampling')([stacked_decoded_parts, theta])
            # outputs.shape == (B, H, W, D, C)
            outputs = tf.reduce_max(resampling_outputs, axis=1)
        else:
            # Composer
            part_decoder = PartDecoder(name='PartDecoder')
            part_decoder_outputs = [part_decoder(x) for x in proj_outputs]
            # stacked_decoded_parts.shape == (B, num_parts, H, W, D, C)
            stacked_decoded_parts = tf.stack(part_decoder_outputs, axis=1)
            # attention_layer_inputs.shape == (B, num_parts, encoding_dimensions)
            attention_layer_inputs = tf.transpose(proj_outputs, (1, 0, 2))
            # attention_outputs.shape == (B, num_parts, 256)
            attention_outputs = AttentionLayer(num_blocks=10, num_heads=8, d_model=100, seq_len=max_num_parts, name='AttentionLayer')(attention_layer_inputs)
            # dense_outputs.shape == (B, num_parts, 12)
            dense_outputs = keras.layers.Dense(12, name='Dense')(attention_outputs)
            # theta.shape == (B, num_parts, 3, 4)
            theta = tf.reshape(dense_outputs, (tf.shape(dense_outputs)[0], max_num_parts, 3, 4))
            # resampling_outputs.shape == (B, num_parts, H, W, D, C)
            resampling_outputs = Resampling(name='Resampling')([stacked_decoded_parts, theta])
            # outputs.shape == (B, H, W, D, C)
            outputs = tf.reduce_max(resampling_outputs, axis=1)
    else:
        # Composer
        part_decoder = PartDecoder(name='PartDecoder')
        part_decoder_outputs = [part_decoder(x) for x in proj_outputs]
        # stacked_decoded_parts.shape == (B, num_part, H, W, D, C)
        stacked_decoded_parts = tf.stack(part_decoder_outputs, axis=1)
        # summed_inputs.shape == (B, encoding_dimensions)
        summed_inputs = tf.reduce_sum(proj_outputs, axis=0)
        # localization_outputs.shape == (B, num_parts, 3, 4)
        localization_outputs = LocalizationNet(name='LocalizationNet', num_parts=max_num_parts)([stacked_decoded_parts, summed_inputs])
        # resampling_outputs.shape == (B, num_parts, H, W, D, C)
        resampling_outputs = Resampling(name='Resampling')([stacked_decoded_parts, localization_outputs])
        # outputs.shape == (B, H, W, D, C)
        outputs = tf.reduce_max(resampling_outputs, axis=1)

    model = keras.Model(inputs, outputs, name=f'{category}')
    return model


class BinaryShapeEncoder(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(BinaryShapeEncoder, self).__init__(**kwargs)

        self.conv1 = layers.Conv3D(16, 5, (1, 1, 1), padding='same')
        self.conv2 = layers.Conv3D(32, 5, (2, 2, 2), padding='same')
        self.conv3 = layers.Conv3D(64, 5, (2, 2, 2), padding='same')
        self.conv4 = layers.Conv3D(128, 3, (2, 2, 2), padding='same')
        self.conv5 = layers.Conv3D(256, 3, (2, 2, 2), padding='same')

        self.act1 = layers.LeakyReLU()
        self.act2 = layers.LeakyReLU()
        self.act3 = layers.LeakyReLU()
        self.act4 = layers.LeakyReLU()
        self.act5 = layers.LeakyReLU()

        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()
        self.bn4 = layers.BatchNormalization()
        self.bn5 = layers.BatchNormalization()

        self.flat = layers.Flatten()
        self.fc = layers.Dense(100, use_bias=False)

    def call(self, inputs, training=False):

        # inputs.shape == (B, H, W, D, C)
        x = self.conv1(inputs)
        x = self.act1(x)
        x = self.bn1(x, training=training)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn2(x, training=training)

        x = self.conv3(x)
        x = self.act3(x)
        x = self.bn3(x, training=training)

        x = self.conv4(x)
        x = self.act4(x)
        x = self.bn4(x, training=training)

        x = self.conv5(x)
        x = self.act5(x)
        x = self.bn5(x, training=training)

        x = self.flat(x)
        # outputs.shape == (B, encoding_dimensions)
        outputs = self.fc(x)

        return outputs


class Projection(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(Projection, self).__init__(**kwargs)
        self.fc = layers.Dense(100, use_bias=False)

    def call(self, inputs):
        # inputs.shape == (B, encoding_dimensions)
        outputs = self.fc(inputs)
        return outputs


class PartDecoder(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(PartDecoder, self).__init__(**kwargs)

        self.fc = layers.Dense(2*2*2*256)
        self.reshape = layers.Reshape((2, 2, 2, 256))

        self.deconv1 = layers.Conv3DTranspose(128, 3, (2, 2, 2), padding='same', output_padding=(1, 1, 1))
        self.deconv2 = layers.Conv3DTranspose(64, 3, (2, 2, 2), padding='same', output_padding=(1, 1, 1))
        self.deconv3 = layers.Conv3DTranspose(32, 5, (2, 2, 2), padding='same', output_padding=(1, 1, 1))
        self.deconv4 = layers.Conv3DTranspose(16, 5, (1, 1, 1), padding='same', output_padding=(0, 0, 0))
        self.deconv5 = layers.Conv3DTranspose(1, 5, (2, 2, 2), padding='same', output_padding=(1, 1, 1), activation='sigmoid')

        self.act = layers.LeakyReLU()
        self.act1 = layers.LeakyReLU()
        self.act2 = layers.LeakyReLU()
        self.act3 = layers.LeakyReLU()
        self.act4 = layers.LeakyReLU()

        self.bn = layers.BatchNormalization()
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()
        self.bn4 = layers.BatchNormalization()

        self.dropout = layers.Dropout(0.2)
        self.dropout1 = layers.Dropout(0.2)
        self.dropout2 = layers.Dropout(0.2)
        self.dropout3 = layers.Dropout(0.2)
        self.dropout4 = layers.Dropout(0.2)

    def call(self, inputs, training=False):

        # inputs.shape == (batch_size, encoding_dimension)
        x = self.fc(inputs)
        x = self.act(x)
        x = self.bn(x, training=training)
        x = self.dropout(x, training=training)
        x = self.reshape(x)

        # x.shape == (B, H, W, D, C)
        x = self.deconv1(x)
        x = self.act1(x)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)

        x = self.deconv2(x)
        x = self.act2(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)

        x = self.deconv3(x)
        x = self.act3(x)
        x = self.bn3(x, training=training)
        x = self.dropout3(x, training=training)

        x = self.deconv4(x)
        x = self.act4(x)
        x = self.bn4(x, training=training)
        x = self.dropout4(x, training=training)

        # outputs.shape == (B, H, W, D, C)
        outputs = self.deconv5(x)

        return outputs


class LocalizationNet(keras.layers.Layer):

    def __init__(self, num_parts, **kwargs):
        super(LocalizationNet, self).__init__(**kwargs)
        self.num_parts = num_parts

        self.stacked_flat = layers.Flatten()

        self.stacked_fc1 = layers.Dense(256)
        self.stacked_fc2 = layers.Dense(256)
        self.summed_fc1 = layers.Dense(128)
        self.final_fc1 = layers.Dense(128)
        self.final_fc2 = layers.Dense(num_parts*12)

        self.stacked_act1 = layers.LeakyReLU()
        self.stacked_act2 = layers.LeakyReLU()
        self.summed_act1 = layers.LeakyReLU()
        self.final_act1 = layers.LeakyReLU()

        self.stacked_dropout1 = layers.Dropout(0.3)
        self.stacked_dropout2 = layers.Dropout(0.3)
        self.summed_dropout1 = layers.Dropout(0.3)
        self.final_dropout1 = layers.Dropout(0.3)

        self.reshape = layers.Reshape((num_parts, 3, 4))

    def call(self, inputs, training=False):

        # stacked_input.shape == (B, num_parts, H, W, D, C)
        stacked_inputs = inputs[0]
        # summed_input.shape == (B, encoding_dimensions)
        summed_inputs = inputs[1]

        # processing stacked inputs
        stacked_x = self.stacked_flat(stacked_inputs)
        stacked_x = self.stacked_fc1(stacked_x)
        stacked_x = self.stacked_act1(stacked_x)
        stacked_x = self.stacked_dropout1(stacked_x, training=training)
        stacked_x = self.stacked_fc2(stacked_x)
        stacked_x = self.stacked_act2(stacked_x)
        stacked_x = self.stacked_dropout2(stacked_x, training=training)

        # processing summed inputs
        summed_x = self.summed_fc1(summed_inputs)
        summed_x = self.summed_act1(summed_x)
        summed_x = self.summed_dropout1(summed_x, training=training)

        # concatenate stacked inputs and summed inputs into final inputs
        final_x = tf.concat([stacked_x, summed_x], axis=1)
        final_x = self.final_fc1(final_x)
        final_x = self.final_act1(final_x)
        final_x = self.final_dropout1(final_x, training=training)
        final_x = self.final_fc2(final_x)

        # outputs.shape == (B, num_parts, 3, 4)
        outputs = self.reshape(final_x)
        return outputs

    def get_config(self):
        config = super().get_config().copy()
        config.update({'num_parts': self.num_parts})
        return config


class Resampling(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(Resampling, self).__init__(**kwargs)

    def call(self, inputs):

        # input_fmap.shape == (B, num_parts, H, W, D, C)
        input_fmap = inputs[0]
        # theta.shape == (B, num_parts, 3, 4)
        theta = inputs[1]

        # batch_grids.shape == (B, num_parts, 3, H, W, D)
        batch_grids = self._affine_grid_generator(input_fmap, theta)

        # x_s.shape == y_s.shape == z_s.shape == (B, num_parts, H, W, D)
        x_s = batch_grids[:, :, 0, :, :, :]
        y_s = batch_grids[:, :, 1, :, :, :]
        z_s = batch_grids[:, :, 2, :, :, :]

        # output_fmap.shape == (B, num_parts, H, W, D, C)
        output_fmap = self._trilinear_sampler(input_fmap, x_s, y_s, z_s)

        return output_fmap

    @staticmethod
    def _affine_grid_generator(input_fmap, theta):
        """
        :param: input_fmap: the stacked decoded parts in the shape of (B, num_parts, H, W, D, C)
        :param: theta: the output of LocalizationNet. has shape (B, num_parts, 3, 4)
        :return: affine grid for the input feature map. affine grid has shape (B, num_parts, 3, H, W, D)
        """

        B, num_parts, H, W, D, C = input_fmap.shape

        # create 3D grid, which are the x, y, z coordinates of the output feature map
        x = tf.cast(tf.linspace(0, W-1, W), dtype=tf.uint8)
        y = tf.cast(tf.linspace(0, H-1, H), dtype=tf.uint8)
        z = tf.cast(tf.linspace(0, D-1, D), dtype=tf.uint8)
        x_t, y_t, z_t = tf.meshgrid(x, y, z)

        # flatten every x, y, z coordinates
        x_t_flat = tf.reshape(x_t, [-1])
        y_t_flat = tf.reshape(y_t, [-1])
        z_t_flat = tf.reshape(z_t, [-1])
        # x_t_flat.shape == (H*W*D,)

        # reshape to (x_t, y_t, z_t, 1), which is homogeneous form
        ones = tf.ones_like(x_t_flat, dtype=tf.uint8)
        sampling_grid = tf.stack([x_t_flat, y_t_flat, z_t_flat, ones])
        # sampling_grid.shape == (4, H*W*D)

        # repeat the grid num_batch times along axis=0 and num_parts times along axis=1
        sampling_grid = tf.expand_dims(sampling_grid, axis=0)
        sampling_grid = tf.expand_dims(sampling_grid, axis=1)
        sampling_grid = tf.tile(sampling_grid, [tf.shape(input_fmap)[0], num_parts, 1, 1])
        # sampling_grid.shape == (B, num_parts, 4, H*W*D)

        # cast to float32 (required for matmul)
        theta = tf.cast(theta, 'float32')
        sampling_grid = tf.cast(sampling_grid, 'float32')

        # transform the sampling grid using batch multiply
        batch_grids = theta @ sampling_grid
        # batch_grids.shape == (B, num_parts, 3, H*W*D)

        # reshape to (B, num_parts, 3, H, W, D)
        batch_grids = tf.reshape(batch_grids, [tf.shape(input_fmap)[0], num_parts, 3, H, W, D])

        return batch_grids

    def _trilinear_sampler(self, input_fmap, x, y, z):

        """
        :param: input_fmap: the stacked decoded parts in the shape of (B, num_parts, H, W, D, C)
        :param: x: x coordinate of input_fmap in the shape of (B, num_parts, H, W, D)
        :param: y: y coordinate of input_fmap in the shape of (B, num_parts, H, W, D)
        :param: z: z coordinate of input_fmap in the shape of (B, num_parts, H, W, D)
        :return: interpolated volume in the shape of (B, num_parts, H, W, D, C)
        """

        # grab 8 nearest corner points for each (x_i, y_i, z_i) in input_fmap
        # 2*2*2 combination, so that there are 8 corner points in total
        B, num_parts, H, W, D, C = input_fmap.shape
        padded_input_fmap = tf.pad(input_fmap, paddings=[[0, 0], [0, 0], [2, 2], [2, 2], [2, 2], [0, 0]])

        x = x + 2
        y = y + 2
        z = z + 2

        x0 = tf.cast(tf.floor(x), 'int32')
        y0 = tf.cast(tf.floor(y), 'int32')
        z0 = tf.cast(tf.floor(z), 'int32')
        x0 = tf.clip_by_value(x0, 0, H + 2)
        y0 = tf.clip_by_value(y0, 0, H + 2)
        z0 = tf.clip_by_value(z0, 0, H + 2)

        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

        # get voxel value of corner coordinates
        # reference: https://blog.csdn.net/webzhuce/article/details/86585489
        c000 = self._get_voxel_value(padded_input_fmap, x0, y0, z0)
        c001 = self._get_voxel_value(padded_input_fmap, x0, y0, z1)
        c010 = self._get_voxel_value(padded_input_fmap, x0, y1, z0)
        c011 = self._get_voxel_value(padded_input_fmap, x0, y1, z1)
        c100 = self._get_voxel_value(padded_input_fmap, x1, y0, z0)
        c101 = self._get_voxel_value(padded_input_fmap, x1, y0, z1)
        c110 = self._get_voxel_value(padded_input_fmap, x1, y1, z0)
        c111 = self._get_voxel_value(padded_input_fmap, x1, y1, z1)
        # cxxx has shape (B, num_parts, H, W, D, C)

        # recast as float for delta calculation
        x0 = tf.cast(x0, 'float32')
        y0 = tf.cast(y0, 'float32')
        z0 = tf.cast(z0, 'float32')

        # calculate deltas
        xd = x - x0
        yd = y - y0
        zd = z - z0

        # compute output
        xd = tf.expand_dims(xd, axis=5)
        yd = tf.expand_dims(yd, axis=5)
        zd = tf.expand_dims(zd, axis=5)
        xd = tf.tile(xd, [1, 1, 1, 1, 1, C])
        yd = tf.tile(yd, [1, 1, 1, 1, 1, C])
        zd = tf.tile(zd, [1, 1, 1, 1, 1, C])
        output_fmap = c000 * (1 - xd) * (1 - yd) * (1 - zd) + c100 * xd * (1 - yd) * (1 - zd) + \
                      c010 * (1 - xd) * yd * (1 - zd) + c001 * (1 - xd) * (1 - yd) * zd + \
                      c101 * xd * (1 - yd) * zd + c011 * (1 - xd) * yd * zd + \
                      c110 * xd * yd * (1 - zd) + c111 * xd * yd * zd
        # output_fmap.shape == (B, num_parts, H, W, D, C)

        return output_fmap

    @staticmethod
    def _get_voxel_value(input_fmap, x, y, z):

        """
        :param: input_fmap: input feature map in the shape of (B, num_parts, H, W, D, C)
        :param: x: x coordinates in the shape of (B, num_parts, H, W, D)
        :param: y: y coordinates in the shape of (B, num_parts, H, W, D)
        :param: z: z coordinates in the shape of (B, num_parts, H, W, D)
        :return: voxel value in the shape of (B, num_parts, H, W, D, C)
        """

        indices = tf.stack([x, y, z], axis=5)
        return tf.gather_nd(input_fmap, indices, batch_dims=2)


class MultiHeadAttention(keras.layers.Layer):

    def __init__(self, num_heads, d_model, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)

        self.num_heads = num_heads
        assert 256 % num_heads == 0
        self.depth = 256 // num_heads

        self.wq = layers.Dense(256)
        self.wk = layers.Dense(256)
        self.wv = layers.Dense(256)
        self.dense = layers.Dense(d_model)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        q = self.wq(inputs)
        k = self.wk(inputs)
        v = self.wv(inputs)
        q = self._split_heads(q, batch_size)
        k = self._split_heads(k, batch_size)
        v = self._split_heads(v, batch_size)

        # scaled_attention.shape == (batch_size, num_heads, seq_len, depth)
        scaled_attention = self._scaled_dot_product_attention(q, k, v)
        # scaled_attention.shape == (batch_size, seq_len, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, (0, 2, 1, 3))
        # concat_attention.shape == (batch_size, seq_len, 256)
        concat_attention = tf.reshape(scaled_attention, (batch_size, seq_len, 256))
        # outputs.shape == (batch_size, seq_len, d_model)
        outputs = self.dense(concat_attention)
        return outputs

    @staticmethod
    def _scaled_dot_product_attention(q, k, v):
        # q.shape == k.shape == (batch_size, num_heads, seq_len, depth)
        # matmul_qk == (batch_size, num_heads, seq_len, seq_len)
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        # softmax is normalized on the last axis so that the scores add up to 1.
        # attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        # outputs.shape == (batch_size, num_heads, seq_len, depth)
        outputs = tf.matmul(attention_weights, v)
        return outputs

    def _split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, (0, 2, 1, 3))


class FeedForward(keras.layers.Layer):

    def __init__(self, d_model, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.dense1 = keras.layers.Dense(256, activation='relu')
        self.dense2 = keras.layers.Dense(d_model)

    def call(self, inputs):
        # x.shape == (batch_size, seq_len, 256)
        x = self.dense1(inputs)
        # outputs.shape == (batch_size, seq_len, d_model)
        outputs = self.dense2(x)
        return outputs


class AttentionBlock(keras.layers.Layer):

    def __init__(self, num_heads, d_model, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.mha = MultiHeadAttention(num_heads, d_model)
        # self.ff = FeedForward(d_model)
        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        # self.layer_norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(0.2)
        # self.dropout2 = keras.layers.Dropout(0.2)

    def call(self, inputs, training=False):
        x = self.mha(inputs)
        x = self.dropout1(x, training=training)
        outputs1 = self.layer_norm1(x + inputs, training=training)
        return outputs1
        # x = self.ff(outputs1)
        # x = self.dropout2(x, training=training)
        ## outputs2.shape == (batch_size, seq_len, d_model)
        # outputs2 = self.layer_norm2(x + outputs1, training=training)
        # return outputs2


class AttentionLayer(keras.layers.Layer):

    def __init__(self, num_blocks, num_heads, d_model, seq_len, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.d_model = d_model
        self.seq_len = seq_len
        self.positional_encoding = self._get_positional_encoding(seq_len, d_model)
        self.attention_blocks = [AttentionBlock(num_heads, d_model) for _ in range(num_blocks)]
        self.dense = keras.layers.Dense(256)

    def call(self, inputs, training=False):
        # inputs.shape == (batch_size, seq_len, d_model)
        # positional_encoding.shape == (1, seq_len, d_model)
        x = inputs + self.positional_encoding
        for each_block in self.attention_blocks:
            x = each_block(x, training=training)
        # outputs.shape == (batch_size, seq_len, 256)
        outputs = self.dense(x)
        return outputs

    @staticmethod
    def _get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def _get_positional_encoding(self, seq_len, d_model):
        angle_rads = self._get_angles(np.arange(seq_len)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'num_blocks': self.num_blocks, 'num_heads': self.num_heads, 'd_model': self.d_model, 'seq_len': self.seq_len})
        return config


if __name__ == '__main__':
    my_model = get_model(attention=True, multi_inputs=True)
    for layer in my_model.layers:
        print(layer.name)

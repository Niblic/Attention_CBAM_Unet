
'''
Squeeze and excitation UNet CBAM


Dependencies:
    Tensorflow 2.16

'''

import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K

# input data
INPUT_SIZE = 512
INPUT_CHANNEL = 9 # 1-grayscale, 4-RGB scale
OUTPUT_MASK_CHANNEL = 1


NUM_FILTER = 32
FILTER_SIZE = 3
UP_SAMP_SIZE = 2

def channel_attention(input_feature, ratio=8):
    """
    Channel-Attention Module (Channel Attention Module)
    """
    shape = tf.shape(input_feature)
    batch_size = shape[0]
    _, _, _, num_channels = input_feature.shape # We still need num_channels from the static shape

    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    max_pool = layers.GlobalMaxPooling2D()(input_feature)

    # Common MLP
    shared_MLP = tf.keras.Sequential([
        layers.Dense(num_channels // ratio, activation='relu', use_bias=False),
        layers.Dense(num_channels, use_bias=False)
    ])

    avg_out = shared_MLP(avg_pool)
    max_out = shared_MLP(max_pool)

    channel_attention_weight = layers.Add()([avg_out, max_out])
    channel_attention_weight = layers.Activation('sigmoid')(channel_attention_weight)
    channel_attention_weight = tf.reshape(channel_attention_weight, (batch_size, 1, 1, num_channels))

    return input_feature * channel_attention_weight

def spatial_attention(input_feature):
    """
    Space  Attention Module (Spatial Attention Module)
    """
    avg_pool = tf.reduce_mean(input_feature, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(input_feature, axis=-1, keepdims=True)

    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])

    spatial_attention_weight = layers.Conv2D(filters=1,
                                             kernel_size=7, # Typische Kernelgröße
                                             strides=1,
                                             padding='same',
                                             activation='sigmoid',
                                             use_bias=False)(concat)

    return input_feature * spatial_attention_weight

def cbam_block(input_feature, ratio=8):
    """
    CBAM-Block (Convolutional Block Attention Module)
    """
    cbam_feature = channel_attention(input_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature
    
def Attention_ResUNet_CBAM(input_shape=( INPUT_SIZE , INPUT_SIZE , INPUT_CHANNEL )):
    inputs = tf.keras.Input(input_shape)

    # Encoder-Pfad
    c1 = conv_block(inputs, NUM_FILTER)
    p1 = layers.MaxPooling2D((UP_SAMP_SIZE, UP_SAMP_SIZE), padding="same")(c1)

    c2 = conv_block(p1, NUM_FILTER * 2)
    p2 = layers.MaxPooling2D((UP_SAMP_SIZE, UP_SAMP_SIZE), padding="same")(c2)

    c3 = conv_block(p2, NUM_FILTER * 4)
    p3 = layers.MaxPooling2D((UP_SAMP_SIZE, UP_SAMP_SIZE), padding="same")(c3)

    c4 = conv_block(p3, NUM_FILTER * 8)
    p4 = layers.MaxPooling2D((UP_SAMP_SIZE, UP_SAMP_SIZE), padding="same")(c4)

    c5 = conv_block(p4, NUM_FILTER * 16)
    p5 = layers.MaxPooling2D((UP_SAMP_SIZE, UP_SAMP_SIZE), padding="same")(c5)
    # Bottleneck
    c6 = conv_block(p5, NUM_FILTER * 32)
    gating = gating_signal(c6, NUM_FILTER * 32)

    # Decoder-Pfad mit CBAM-Blocks und Skip-Connections
    u7 = upsample_block(gating, NUM_FILTER * 16)
    cbam7 = cbam_block(u7)
    c7 = conv_block(layers.concatenate([cbam7, c5]), NUM_FILTER * 16)

    u8 = upsample_block(c7, NUM_FILTER * 8)
    cbam8 = cbam_block(u8)
    c8 = conv_block(layers.concatenate([cbam8, c4]), NUM_FILTER * 8)

    u9 = upsample_block(c8, NUM_FILTER * 4)
    cbam9 = cbam_block(u9)
    c9 = conv_block(layers.concatenate([cbam9, c3]), NUM_FILTER * 4)

    u10 = upsample_block(c9, NUM_FILTER * 2)
    cbam10 = cbam_block(u10)
    c10 = conv_block(layers.concatenate([cbam10, c2]), NUM_FILTER * 2)

    u11 = upsample_block(c10, NUM_FILTER)
    cbam11 = cbam_block(u11)
    c11 = conv_block(layers.concatenate([cbam11, c1]), NUM_FILTER)
    # Ausgabeschicht
    conv_final = layers.Conv2D(OUTPUT_MASK_CHANNEL, (1, 1), activation="sigmoid")(c11)

    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('sigmoid')(conv_final)

    model = models.Model(inputs=[inputs], outputs=conv_final , name="Attention_ResUNet_CBAM")
    return model    
    

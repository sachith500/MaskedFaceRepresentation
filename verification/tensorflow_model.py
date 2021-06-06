import tensorflow as tf
import tensorflow.keras.backend as K
from keras_vggface.vggface import VGGFace
from tensorflow.python.keras import *
from tensorflow.python.keras.layers import *

from verification.config import IMG_SHAPE


def square_change(vectors):
    [feats_a, feats_b] = vectors
    square = K.square((feats_a - feats_b))

    return square


class TensorflowSiameseNetworkUsingSenet:
    def __init__(self, model_path):
        self.fine_tune_percentage = 0.7
        self.model = self.build_siamese_network_model(self.fine_tune_percentage)
        self.model_path = model_path

    def build(self):
        return self.model

    def build_siamese_network_model(self, fine_tune_percentage):
        input_a = tf.keras.layers.Input(shape=IMG_SHAPE)
        input_b = tf.keras.layers.Input(shape=IMG_SHAPE)

        sister_network = self.build_network(IMG_SHAPE, True, fine_tune_percentage)

        feature_a = sister_network(input_a)
        feature_b = sister_network(input_b)

        diff = tf.keras.layers.Lambda(square_change)([feature_a, feature_b])

        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(diff)
        model = tf.keras.Model(inputs=[input_a, input_b], outputs=outputs)

        optimizer = tf.keras.optimizers.SGD(lr=0.001)
        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        return model

    def build_network(self, shape, finetune=False, fine_tune_percentage=.30):
        base_model = VGGFace(model='senet50', include_top=False, input_shape=shape)
        if not finetune:
            base_model.trainable = False
        else:
            base_model.trainable = True
            # Fine-tune from this layer onwards
            fine_tune_at = 272
            # Freeze all the layers before the `fine_tune_at` layer
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
            for layer in base_model.layers[fine_tune_at:]:
                layer.trainable = True
        last_layer = base_model.get_layer('avg_pool').output
        x = Flatten(name='flatten')(last_layer)
        out = Dense(512, activation='relu', name='fc7')(x)
        model = Model(base_model.input, out)
        return model


class TensorflowSiameseNetworkUsingVgg16:
    def __init__(self, model_path):
        self.model_path = model_path
        self.fine_tune_percentage = 0.01
        self.model = self.build_siamese_network_model(self.fine_tune_percentage)
        self.model.load_weights(model_path)

    def build(self):
        return self.model

    def build_siamese_network_model(self, fine_tune_percentage):
        input_a = tf.keras.layers.Input(shape=IMG_SHAPE)
        input_b = tf.keras.layers.Input(shape=IMG_SHAPE)

        sister_network = self.build_network(IMG_SHAPE, True, fine_tune_percentage)

        feature_a = sister_network(input_a)
        feature_b = sister_network(input_b)

        diff = tf.keras.layers.Lambda(square_change)([feature_a, feature_b])

        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(diff)
        model = tf.keras.Model(inputs=[input_a, input_b], outputs=outputs)

        return model

    def build_network(self, shape, finetune=False, fine_tune_percentage=.30):
        base_model = VGGFace(model='vgg16', include_top=False, input_shape=shape, pooling="avg")
        if not finetune:
            base_model.trainable = False
        else:
            base_model.trainable = True
            fine_tune_at = 17
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
            for layer in base_model.layers[fine_tune_at:]:
                layer.trainable = True
        # last_layer = base_model.get_layer('avg_pool').output
        # x = Flatten(name='flatten')(last_layer)
        out = Dense(512, activation='relu', name='fc7')(base_model.output)
        model = Model(base_model.input, out)
        return model


class TensorflowSiameseNetworkUsingVgg19:
    def __init__(self, model_path):
        self.fine_tune_percentage = 0.6
        self.model_path = model_path

    def build(self):
        sister_fc_network_base_weights = f"./models/benchmark_1/sis_fc_network_vgg19.h5"
        backbone_network_base_weights = f"./models/benchmark_1/backbone_vgg19.h5"
        distance_network_base_weights = f"./models/benchmark_1/distance_network_vgg19.h5"
        model, back_bone, distance_network, sister_fc_model = self.build_siamese_network(backbone_network_base_weights)
        sister_fc_model.load_weights(sister_fc_network_base_weights)
        distance_network.load_weights(distance_network_base_weights)

        return model

    def build_siamese_network(self, backbone_network_base_weights):
        input_a = tf.keras.layers.Input(shape=IMG_SHAPE)
        input_b = tf.keras.layers.Input(shape=IMG_SHAPE)

        sister_network, back_bone, sister_fc_model = self.build_vgg_19_sister_network(IMG_SHAPE,
                                                                                      backbone_network_base_weights)
        distance_network = self.build_distance_model(512)

        feature_a = sister_network(input_a)
        feature_b = sister_network(input_b)
        distance = tf.keras.layers.Lambda(square_change)([feature_a, feature_b])

        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(distance)
        model = tf.keras.Model(inputs=[input_a, input_b], outputs=outputs)

        optimizer = tf.keras.optimizers.SGD(lr=0.0001)
        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"], )

        return model, back_bone, distance_network, sister_fc_model

    def build_vgg_19_sister_network(self, shape, base_weights):
        inputs = tf.keras.layers.Input(shape)
        base_model = tf.keras.applications.vgg19.VGG19(input_shape=shape, include_top=False, weights=base_weights)
        sister_fc_model = self.build_sister_fc_model((7, 7, 512))
        x = base_model(inputs)
        outputs = sister_fc_model(x)
        model = tf.keras.Model(inputs, outputs)
        return model, base_model, sister_fc_model

    def build_sister_fc_model(self, shape):
        inputs = tf.keras.layers.Input(shape)

        outputs = tf.keras.layers.GlobalAveragePooling2D()(inputs)

        model = tf.keras.Model(inputs, outputs)

        return model

    def build_distance_model(self, shape):
        inputs = tf.keras.layers.Input(shape)

        fc1 = tf.keras.layers.Dense(256, activation="relu")(inputs)
        fc2 = tf.keras.layers.Dense(64, activation="relu")(fc1)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(fc2)

        model = tf.keras.Model(inputs, outputs)

        return model


class TensorflowSiameseNetworkUsingMobilenet:
    def __init__(self, model_path):
        self.fine_tune_percentage = 0.3
        self.model = self.build_siamese_model(self.fine_tune_percentage)
        self.model.load_weights(model_path)

    def build(self):
        return self.model

    def build_siamese_model(self, fine_tune_percentage):
        input_a = tf.keras.layers.Input(shape=IMG_SHAPE)
        input_b = tf.keras.layers.Input(shape=IMG_SHAPE)

        sister_network = self.build_Mobile_sister_network(IMG_SHAPE, True, fine_tune_percentage)

        feature_a = sister_network(input_a)
        feature_b = sister_network(input_b)

        diff = tf.keras.layers.Lambda(square_change)([feature_a, feature_b])

        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(diff)
        model = tf.keras.Model(inputs=[input_a, input_b], outputs=outputs)
        return model

    def build_Mobile_sister_network(self, shape, fineTune=False, fine_tune_percentage=.30):
        inputs = tf.keras.layers.Input(shape)
        base_model = tf.keras.applications.MobileNetV2(input_shape=shape, include_top=False, weights="imagenet", )
        if fineTune == False:
            base_model.trainable = False
        else:
            base_model.trainable = True
            # Fine-tune from this layer onwards
            fine_tune_at = len(base_model.layers) - int(len(base_model.layers) * fine_tune_percentage)
            # Freeze all the layers before the `fine_tune_at` layer
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
            for layer in base_model.layers[fine_tune_at:]:
                layer.trainable = True
        x = base_model(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        outputs = tf.keras.layers.Dropout(0.2)(x)

        model = tf.keras.Model(inputs, outputs)
        return model

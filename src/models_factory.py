"""Model factory for Transfer Learning backbones, classification heads, and baseline CNN."""

from __future__ import annotations

import keras
from keras import layers

from src.config import (
    DEFAULT_LEARNING_RATE,
    DENSE_UNITS,
    MODEL_SPECS,
    ModelSpec,
)


def _apply_backbone_preprocessing(
    inputs: keras.KerasTensor,
    model_key: str,
    data_format: str = "channels_last",
) -> keras.KerasTensor:
    """Apply native Keras 3 serializable preprocessing matching each ImageNet backbone."""
    if model_key == "vgg16":
        return keras.applications.vgg16.preprocess_input(inputs, data_format=data_format)
    if model_key == "resnet50":
        return keras.applications.resnet50.preprocess_input(inputs, data_format=data_format)
    if model_key in ("inception_v3", "mobilenet_v2"):
        return layers.Rescaling(
            scale=1.0 / 127.5, offset=-1.0, name=f"{model_key}_rescale"
        )(inputs)
    raise ValueError(f"Unsupported model_key: {model_key}")


def _load_frozen_backbone(
    model_key: str,
    spec: ModelSpec,
    data_format: str = "channels_last",
) -> keras.Model:
    """Instantiate an ImageNet-pretrained convolutional base with frozen weights."""
    h, w = spec.image_size
    input_shape = (3, h, w) if data_format == "channels_first" else (h, w, 3)
    backbone_cls_map = {
        "vgg16": keras.applications.VGG16,
        "resnet50": keras.applications.ResNet50,
        "inception_v3": keras.applications.InceptionV3,
        "mobilenet_v2": keras.applications.MobileNetV2,
    }
    if model_key not in backbone_cls_map:
        raise ValueError(f"Unsupported model_key: {model_key}")

    backbone = backbone_cls_map[model_key](
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    backbone.trainable = False
    return backbone


def build_feature_extractor(
    model_key: str,
    data_format: str = "channels_last",
) -> tuple[keras.Model, keras.Model]:
    """Build a frozen feature extractor (RGB [0,255] -> preprocess -> backbone -> GAP).

    Returns:
        (extractor_model, frozen_backbone)
    """
    spec = MODEL_SPECS[model_key]
    h, w = spec.image_size
    in_shape = (3, h, w) if data_format == "channels_first" else (h, w, 3)
    inputs = keras.Input(shape=in_shape, name="input_rgb")
    x = _apply_backbone_preprocessing(inputs, model_key, data_format=data_format)
    backbone = _load_frozen_backbone(model_key, spec, data_format=data_format)
    # training=False keeps BatchNormalization statistics in inference mode
    x = backbone(x, training=False)
    features = layers.GlobalAveragePooling2D(
        data_format=data_format, name="global_avg_pool"
    )(x)
    extractor = keras.Model(
        inputs=inputs,
        outputs=features,
        name=f"{model_key}_feature_extractor",
    )
    return extractor, backbone


def build_classifier_head(
    feature_dim: int,
    dropout_rate: float = 0.5,
    dense_units: int = DENSE_UNITS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    name: str = "classifier_head",
) -> keras.Model:
    """Build and compile the trainable classification head on top of bottleneck features."""
    feat_input = keras.Input(shape=(feature_dim,), name="bottleneck_features")
    x = layers.Dense(dense_units, activation="relu", name="fc_dense")(feat_input)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="predictions")(x)

    head = keras.Model(inputs=feat_input, outputs=outputs, name=name)
    head.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return head


def assemble_end_to_end_model(
    model_key: str,
    trained_head: keras.Model,
    backbone: keras.Model | None = None,
    dropout_rate: float = 0.5,
    dense_units: int = DENSE_UNITS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> keras.Model:
    """Stitch the frozen ImageNet backbone and trained head into a standalone end-to-end Keras model."""
    spec = MODEL_SPECS[model_key]
    if backbone is None:
        backbone = _load_frozen_backbone(model_key, spec)
    backbone.trainable = False

    inputs = keras.Input(shape=(*spec.image_size, 3), name="input_rgb")
    x = _apply_backbone_preprocessing(inputs, model_key)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    fc_layer = layers.Dense(dense_units, activation="relu", name="fc_dense")
    drop_layer = layers.Dropout(dropout_rate, name="dropout")
    pred_layer = layers.Dense(1, activation="sigmoid", name="predictions")

    x = fc_layer(x)
    x = drop_layer(x)
    outputs = pred_layer(x)

    full_model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"{model_key}_transfer_learning",
    )

    # Copy learned weights from trained_head into the end-to-end classification layers
    fc_layer.set_weights(trained_head.get_layer("fc_dense").get_weights())
    pred_layer.set_weights(trained_head.get_layer("predictions").get_weights())

    full_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return full_model


def build_scratch_cnn_baseline(
    image_size: tuple[int, int] = (150, 150),
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> keras.Model:
    """Build a classic 4-block CNN from scratch (Lab 7 reference without Transfer Learning)."""
    inputs = keras.Input(shape=(*image_size, 3), name="input_rgb")
    x = layers.Rescaling(1.0 / 255.0, name="rescale_0_1")(inputs)

    for idx, filters in enumerate((32, 64, 128, 128), start=1):
        x = layers.Conv2D(filters, (3, 3), activation="relu", padding="same", name=f"conv_{idx}")(x)
        x = layers.MaxPooling2D((2, 2), name=f"pool_{idx}")(x)

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(256, activation="relu", name="fc_dense")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="scratch_cnn_lab7_baseline")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def count_model_parameters(model: keras.Model) -> dict[str, int]:
    """Compute total, trainable, and non-trainable parameter counts for reporting."""
    trainable = int(sum(keras.ops.size(w) for w in model.trainable_weights))
    non_trainable = int(sum(keras.ops.size(w) for w in model.non_trainable_weights))
    return {
        "total_params": trainable + non_trainable,
        "trainable_params": trainable,
        "non_trainable_params": non_trainable,
    }


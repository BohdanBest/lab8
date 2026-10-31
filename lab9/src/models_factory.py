"""Model factory for Fine-Tuning backbones, configurable heads, prefix/suffix splitting, and End-to-End .keras assembly."""

from __future__ import annotations

from typing import Any

import keras
from keras import layers

from .config import (
    DEFAULT_DENSE_UNITS,
    DEFAULT_DROPOUT,
    DEFAULT_FINETUNE_LR,
    DEFAULT_WARMUP_LR,
    MODEL_SPECS,
    ModelSpec,
)

BACKBONE_CLS_MAP: dict[str, Any] = {
    "vgg16": keras.applications.VGG16,
    "resnet50": keras.applications.ResNet50,
    "inception_v3": keras.applications.InceptionV3,
    "mobilenet_v2": keras.applications.MobileNetV2,
}


def build_optimizer(optimizer_name: str, learning_rate: float) -> keras.Optimizer:
    """Instantiate a Keras optimizer ('adam', 'rmsprop', or 'sgd') with the given learning rate."""
    opt_key = optimizer_name.strip().lower()
    if opt_key == "adam":
        return keras.optimizers.Adam(learning_rate=learning_rate)
    if opt_key == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=learning_rate)
    if opt_key == "sgd":
        return keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Expected 'adam', 'rmsprop', or 'sgd'.")


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


def load_pretrained_backbone(
    model_key: str,
    spec: ModelSpec | None = None,
    data_format: str = "channels_last",
) -> keras.Model:
    """Instantiate an ImageNet-pretrained convolutional base (`include_top=False`)."""
    spec = spec or MODEL_SPECS[model_key]
    h, w = spec.image_size
    input_shape = (3, h, w) if data_format == "channels_first" else (h, w, 3)
    if model_key not in BACKBONE_CLS_MAP:
        raise ValueError(f"Unsupported model_key: {model_key}")

    return BACKBONE_CLS_MAP[model_key](
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )


def configure_backbone_unfreezing(
    backbone: keras.Model,
    cut_layer_name: str | None,
    freeze_batch_norm: bool = True,
) -> dict[str, int]:
    """Freeze all layers up to `cut_layer_name` and unfreeze top convolutional blocks after it.

    When `freeze_batch_norm=True`, all `BatchNormalization` layers inside the unfrozen blocks
    remain `trainable=False` so their ImageNet running statistics are preserved.
    """
    backbone.trainable = True
    if cut_layer_name is None:
        for layer in backbone.layers:
            layer.trainable = False
    else:
        cut_layer = backbone.get_layer(cut_layer_name)
        cut_idx = backbone.layers.index(cut_layer)
        for idx, layer in enumerate(backbone.layers):
            if idx <= cut_idx:
                layer.trainable = False
            elif freeze_batch_norm and isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

    unfrozen_conv_layers = sum(
        1
        for layer in backbone.layers
        if layer.trainable and len(layer.weights) > 0
    )
    frozen_bn_layers = sum(
        1
        for layer in backbone.layers
        if isinstance(layer, layers.BatchNormalization) and not layer.trainable
    )
    return {
        "total_backbone_layers": len(backbone.layers),
        "trainable_weight_layers_in_backbone": unfrozen_conv_layers,
        "frozen_bn_layers": frozen_bn_layers,
    }


def build_feature_extractor(
    model_key: str,
    data_format: str = "channels_last",
) -> tuple[keras.Model, keras.Model]:
    """Build full frozen feature extractor (RGB [0,255] -> preprocess -> backbone -> GAP)."""
    spec = MODEL_SPECS[model_key]
    h, w = spec.image_size
    in_shape = (3, h, w) if data_format == "channels_first" else (h, w, 3)
    inputs = keras.Input(shape=in_shape, name="input_rgb")
    x = _apply_backbone_preprocessing(inputs, model_key, data_format=data_format)
    backbone = load_pretrained_backbone(model_key, spec, data_format=data_format)
    backbone.trainable = False
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


def build_prefix_extractor(
    model_key: str,
    cut_layer_name: str,
    data_format: str = "channels_last",
) -> keras.Model:
    """Build frozen prefix model from RGB [0,255] up to `cut_layer_name` output tensor."""
    spec = MODEL_SPECS[model_key]
    h, w = spec.image_size
    in_shape = (3, h, w) if data_format == "channels_first" else (h, w, 3)

    backbone = load_pretrained_backbone(model_key, spec, data_format=data_format)
    backbone.trainable = False
    cut_layer = backbone.get_layer(cut_layer_name)
    prefix_backbone = keras.Model(
        inputs=backbone.input,
        outputs=cut_layer.output,
        name=f"{model_key}_frozen_prefix_core",
    )
    prefix_backbone.trainable = False

    inputs = keras.Input(shape=in_shape, name="input_rgb")
    x = _apply_backbone_preprocessing(inputs, model_key, data_format=data_format)
    prefix_out = prefix_backbone(x, training=False)
    return keras.Model(
        inputs=inputs,
        outputs=prefix_out,
        name=f"{model_key}_prefix_{cut_layer_name}",
    )


def build_classifier_head(
    feature_dim: int,
    dense_units: int = DEFAULT_DENSE_UNITS,
    dropout_rate: float = DEFAULT_DROPOUT,
    second_dense_units: int = 0,
    optimizer_name: str = "adam",
    learning_rate: float = DEFAULT_WARMUP_LR,
    name: str = "classifier_head",
) -> keras.Model:
    """Build and compile a 1-layer or 2-layer Dense classification head for Stage 1 warm-up."""
    feat_input = keras.Input(shape=(feature_dim,), name="bottleneck_features")
    x = layers.Dense(dense_units, activation="relu", name="fc_dense_1")(feat_input)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)
    if second_dense_units > 0:
        x = layers.Dense(second_dense_units, activation="relu", name="fc_dense_2")(x)
        x = layers.Dropout(min(dropout_rate, 0.5), name="dropout_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="predictions")(x)

    head = keras.Model(inputs=feat_input, outputs=outputs, name=name)
    head.compile(
        optimizer=build_optimizer(optimizer_name, learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return head


def build_trainable_suffix_model(
    model_key: str,
    cut_layer_name: str,
    trained_head: keras.Model | None = None,
    dense_units: int = DEFAULT_DENSE_UNITS,
    dropout_rate: float = DEFAULT_DROPOUT,
    second_dense_units: int = 0,
    optimizer_name: str = "adam",
    learning_rate: float = DEFAULT_FINETUNE_LR,
    freeze_batch_norm: bool = True,
    data_format: str = "channels_last",
) -> tuple[keras.Model, dict[str, int]]:
    """Build the trainable suffix (unfrozen top conv blocks + classification head) after `cut_layer_name`."""
    spec = MODEL_SPECS[model_key]
    backbone = load_pretrained_backbone(model_key, spec, data_format=data_format)
    unfreeze_meta = configure_backbone_unfreezing(
        backbone=backbone,
        cut_layer_name=cut_layer_name,
        freeze_batch_norm=freeze_batch_norm,
    )

    cut_layer = backbone.get_layer(cut_layer_name)
    cut_idx = backbone.layers.index(cut_layer)
    suffix_in = keras.Input(shape=cut_layer.output.shape[1:], name="prefix_features_in")

    tensor_map: dict[int, keras.KerasTensor] = {id(cut_layer.output): suffix_in}
    for layer in backbone.layers[cut_idx + 1 :]:
        in_tensors = layer._inbound_nodes[0].input_tensors
        mapped_in = (
            tensor_map[id(in_tensors[0])]
            if len(in_tensors) == 1
            else [tensor_map[id(t)] for t in in_tensors]
        )
        if freeze_batch_norm and isinstance(layer, layers.BatchNormalization):
            out_t = layer(mapped_in, training=False)
        else:
            out_t = layer(mapped_in)
        tensor_map[id(layer.output)] = out_t

    x = tensor_map[id(backbone.output)]
    x = layers.GlobalAveragePooling2D(data_format=data_format, name="global_avg_pool")(x)

    fc1 = layers.Dense(dense_units, activation="relu", name="fc_dense_1")
    drop1 = layers.Dropout(dropout_rate, name="dropout_1")
    x = drop1(fc1(x))

    fc2: layers.Dense | None = None
    if second_dense_units > 0:
        fc2 = layers.Dense(second_dense_units, activation="relu", name="fc_dense_2")
        drop2 = layers.Dropout(min(dropout_rate, 0.5), name="dropout_2")
        x = drop2(fc2(x))

    pred_layer = layers.Dense(1, activation="sigmoid", name="predictions")
    outputs = pred_layer(x)

    suffix_model = keras.Model(
        inputs=suffix_in,
        outputs=outputs,
        name=f"{model_key}_finetune_suffix",
    )

    # Initialize head weights from Stage 1 warm-up to protect ImageNet conv filters
    if trained_head is not None:
        fc1.set_weights(trained_head.get_layer("fc_dense_1").get_weights())
        if fc2 is not None:
            fc2.set_weights(trained_head.get_layer("fc_dense_2").get_weights())
        pred_layer.set_weights(trained_head.get_layer("predictions").get_weights())

    suffix_model.compile(
        optimizer=build_optimizer(optimizer_name, learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return suffix_model, unfreeze_meta


def assemble_finetuned_end_to_end_model(
    model_key: str,
    cut_layer_name: str,
    trained_suffix_model: keras.Model,
    dense_units: int = DEFAULT_DENSE_UNITS,
    dropout_rate: float = DEFAULT_DROPOUT,
    second_dense_units: int = 0,
    optimizer_name: str = "adam",
    learning_rate: float = DEFAULT_FINETUNE_LR,
    freeze_batch_norm: bool = True,
) -> keras.Model:
    """Stitch frozen prefix + fine-tuned top blocks + head into a universal NHWC FP32 .keras model."""
    spec = MODEL_SPECS[model_key]
    backbone = load_pretrained_backbone(model_key, spec, data_format="channels_last")
    configure_backbone_unfreezing(
        backbone=backbone,
        cut_layer_name=cut_layer_name,
        freeze_batch_norm=freeze_batch_norm,
    )

    # Copy fine-tuned convolutional weights from suffix_model into the full backbone
    suffix_layer_names = {layer.name for layer in trained_suffix_model.layers}
    cut_layer = backbone.get_layer(cut_layer_name)
    cut_idx = backbone.layers.index(cut_layer)
    for layer in backbone.layers[cut_idx + 1 :]:
        if layer.name in suffix_layer_names and len(layer.weights) > 0:
            layer.set_weights(trained_suffix_model.get_layer(layer.name).get_weights())

    inputs = keras.Input(shape=(*spec.image_size, 3), name="input_rgb")
    x = _apply_backbone_preprocessing(inputs, model_key, data_format="channels_last")
    # Pass through backbone functional graph preserving BatchNorm inference mode when frozen
    tensor_map: dict[int, keras.KerasTensor] = {id(backbone.input): x}
    for layer in backbone.layers[1:]:
        in_tensors = layer._inbound_nodes[0].input_tensors
        mapped_in = (
            tensor_map[id(in_tensors[0])]
            if len(in_tensors) == 1
            else [tensor_map[id(t)] for t in in_tensors]
        )
        if freeze_batch_norm and isinstance(layer, layers.BatchNormalization):
            out_t = layer(mapped_in, training=False)
        else:
            out_t = layer(mapped_in)
        tensor_map[id(layer.output)] = out_t

    x = tensor_map[id(backbone.output)]
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    fc1 = layers.Dense(dense_units, activation="relu", name="fc_dense_1")
    drop1 = layers.Dropout(dropout_rate, name="dropout_1")
    x = drop1(fc1(x))

    fc2: layers.Dense | None = None
    if second_dense_units > 0:
        fc2 = layers.Dense(second_dense_units, activation="relu", name="fc_dense_2")
        drop2 = layers.Dropout(min(dropout_rate, 0.5), name="dropout_2")
        x = drop2(fc2(x))

    pred_layer = layers.Dense(1, activation="sigmoid", name="predictions")
    outputs = pred_layer(x)

    full_model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"{model_key}_finetuned",
    )

    fc1.set_weights(trained_suffix_model.get_layer("fc_dense_1").get_weights())
    if fc2 is not None:
        fc2.set_weights(trained_suffix_model.get_layer("fc_dense_2").get_weights())
    pred_layer.set_weights(trained_suffix_model.get_layer("predictions").get_weights())

    # Synchronize layer.trainable flags on the assembled model for accurate parameter reporting
    cut_reached = False
    for layer in full_model.layers:
        if layer.name == cut_layer_name:
            cut_reached = True
            layer.trainable = False
        elif not cut_reached and layer.name not in ("fc_dense_1", "fc_dense_2", "predictions"):
            layer.trainable = False
        elif freeze_batch_norm and isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    full_model.compile(
        optimizer=build_optimizer(optimizer_name, learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return full_model


def count_model_parameters(model: keras.Model) -> dict[str, int]:
    """Compute total, trainable, and non-trainable parameter counts for reporting."""
    trainable = int(sum(keras.ops.size(w) for w in model.trainable_weights))
    non_trainable = int(sum(keras.ops.size(w) for w in model.non_trainable_weights))
    return {
        "total_params": trainable + non_trainable,
        "trainable_params": trainable,
        "non_trainable_params": non_trainable,
    }


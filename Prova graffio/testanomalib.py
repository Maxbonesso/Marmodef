from multiprocessing import freeze_support
from pathlib import Path
from anomalib.data.utils import read_image
from anomalib.deploy import OpenVINOInferencer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from anomalib.data.utils import DownloadInfo, download_and_extract
from anomalib.data.folder import Folder
from anomalib.data.task_type import TaskType
from anomalib.models import Padim
from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.utils.callbacks import (
    MetricsConfigurationCallback,
    MinMaxNormalizationCallback,
    PostProcessingConfigurationCallback,
)
from anomalib.utils.callbacks.export import ExportCallback, ExportMode
from matplotlib import pyplot as plt
from anomalib.post_processing import Visualizer, VisualizationMode
from PIL import Image
if __name__ == "__main__":
    freeze_support()

    dataset_download_info = DownloadInfo(
        name="cubes.zip",
        url="https://github.com/openvinotoolkit/anomalib/releases/download/dobot/cubes.zip",
        hash="e6e067f9e0979a4d190dd2cb1db227d7",
    )
    api_download_info = DownloadInfo(
        name="dobot_api.zip",
        url="https://github.com/openvinotoolkit/anomalib/releases/download/dobot/dobot_api.zip",
        hash="89d6d6400cdff03de3c25d2c54f2b443",
    )
    # download_and_extract(root=Path.cwd(), info=dataset_download_info)
    # download_and_extract(root=Path.cwd(), info=api_download_info)


    datamodule = Folder(
        root=Path.cwd() / "cubes",
        normal_dir="normal",
        abnormal_dir="abnormal",
        normal_split_ratio=0.2,
        image_size=(256, 256),
        train_batch_size=32,
        eval_batch_size=32,
        task=TaskType.CLASSIFICATION,
    )
    datamodule.setup()  # Split the data to train/val/test/prediction sets.
    datamodule.prepare_data()  # Create train/val/test/predic dataloaders

    i, data = next(enumerate(datamodule.val_dataloader()))
    print(data.keys())

    # Check image size
    print(data["image"].shape)


    model = Padim(
        input_size=(256, 256),
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
    )


    callbacks = [
        MetricsConfigurationCallback(
            task=TaskType.CLASSIFICATION,
            image_metrics=["AUROC"],
        ),
        ModelCheckpoint(
            mode="max",
            monitor="image_AUROC",
        ),
        PostProcessingConfigurationCallback(
            normalization_method=NormalizationMethod.MIN_MAX,
            threshold_method=ThresholdMethod.ADAPTIVE,
        ),
        MinMaxNormalizationCallback(),
        ExportCallback(
            input_size=(256, 256),
            dirpath=str(Path.cwd()),
            filename="model",
            export_mode=ExportMode.OPENVINO,
        ),
    ]


    trainer = Trainer(
        callbacks=callbacks,
        accelerator="auto",
        auto_scale_batch_size=False,
        check_val_every_n_epoch=1,
        devices="auto",
        strategy="auto",
        max_epochs=1,
        num_sanity_val_steps=0,
        val_check_interval=1.0,
    )
    trainer.fit(model=model, datamodule=datamodule)


    image_path = "./cubes/abnormal/input_20230210134059.jpg"
    image = read_image(path="./cubes/abnormal/input_20230210134059.jpg")
    plt.imshow(image)


    openvino_model_path = Path.cwd() / "weights" / "openvino" / "model.bin"
    metadata_path = Path.cwd() / "weights" / "openvino" / "metadata.json"
    print(openvino_model_path.exists(), metadata_path.exists())


    inferencer = OpenVINOInferencer(
        path=openvino_model_path,  # Path to the OpenVINO IR model.
        metadata=metadata_path,  # Path to the metadata file.
        device="CPU",  # We would like to run it on an Intel CPU.
    )


    print(image.shape)
    predictions = inferencer.predict(image=image)


    visualizer = Visualizer(mode=VisualizationMode.FULL, task=TaskType.CLASSIFICATION)
    output_image = visualizer.visualize_image(predictions)
    Image.fromarray(output_image)


    visualizer = Visualizer(mode=VisualizationMode.FULL, task=TaskType.SEGMENTATION)
    output_image = visualizer.visualize_image(predictions)
    Image.fromarray(output_image)
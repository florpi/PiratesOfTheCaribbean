import logging

logging.getLogger().setLevel(logging.INFO)
from glob import glob
import numpy as np
import pandas as pd

# re-trying
from tenacity import retry, stop_after_attempt, wait_random

# Geo-processing
import geopandas as gpd
import rasterio
import rasterio.mask
from shapely.geometry import box

# Image processing
import cv2

# Keras
import keras
from keras_preprocessing.image.iterator import Iterator

LABELMAP = {
    "concrete_cement": 0,
    "healthy_metal": 1,
    "incomplete": 2,
    "irregular_metal": 3,
    "other": 4,
}
LABELMAP_INV = {
    0: "concrete_cement",
    1: "healthy_metal",
    2: "incomplete",
    3: "irregular_metal",
    4: "other",
}

# def smooth_labels(y, smooth_factor):
#     """Convert a matrix of one-hot row-vector labels into smoothed versions.
#
#     # Arguments
#         y: matrix of one-hot row-vector labels to be smoothed
#         smooth_factor: label smoothing factor (between 0 and 1)
#
#     # Returns
#         A matrix of smoothed labels.
#     """
#     assert len(y.shape) == 2
#     if 0 <= smooth_factor <= 1:
#         # label smoothing ref: https://www.robots.ox.ac.uk/~vgg/rg/papers/reinception.pdf
#         y *= 1 - smooth_factor
#         y += smooth_factor / y.shape[1]
#     else:
#         raise Exception("Invalid label smoothing factor: " + str(smooth_factor))
#     return y


def smooth_labels(y, smooth_factor):
    """Convert a matrix of one-hot row-vector labels into smoothed versions.

    # Arguments
        y: matrix of one-hot row-vector labels to be smoothed
        smooth_factor: label smoothing factor (between 0 and 1)

    # Returns
        A matrix of smoothed labels.
    ["concrete_cement", "healthy_metal", "incomplete", "irregular_metal", "other"]
    """
    assert len(y.shape) == 1
    if 0 <= smooth_factor <= 1:
        smooth_factor = float(smooth_factor)
        # label smoothing ref: https://www.robots.ox.ac.uk/~vgg/rg/papers/reinception.pdf
        label_id = LABELMAP_INV[np.argmax(y)]
        if label_id == "concrete_cement":
            return [
                1 - smooth_factor,
                smooth_factor / 3,
                smooth_factor / 3,
                smooth_factor / 3,
                0,
            ]
        elif label_id == "healthy_metal":
            return [0, 1 - smooth_factor, 0, smooth_factor, 0]
        elif label_id == "incomplete":
            return [
                smooth_factor / 3,
                smooth_factor / 3,
                1 - smooth_factor,
                smooth_factor / 3,
                0,
            ]
        elif label_id == "irregular_metal":
            return [0, smooth_factor, 0, 1 - smooth_factor, 0]
        else:
            return y
    else:
        raise Exception("Invalid label smoothing factor: " + str(smooth_factor))
    return y


# Image preprocessing
def resize_with_pad(img, height, width):
    """
    Re-adapted numpy/opencv code from original tensorflow function:
        https://www.tensorflow.org/api_docs/python/tf/image/resize_with_pad
    """
    f_target_height = float(height)
    f_target_width = float(width)
    i_height, i_width = img.shape[:2]
    f_height = float(i_height)
    f_width = float(i_width)
    # Find the ratio by which the image must be adjusted
    # to fit within the target
    ratio = max(f_width / f_target_width, f_height / f_target_height)
    resized_height_float = f_height / ratio
    resized_width_float = f_width / ratio
    resized_height = int(resized_height_float)
    resized_width = int(resized_width_float)
    padding_height = (f_target_height - resized_height_float) / 2
    padding_width = (f_target_width - resized_width_float) / 2
    f_padding_height = float(int(padding_height))
    f_padding_width = float(int(padding_width))
    p_height = max(0, int(f_padding_height))
    p_width = max(0, int(f_padding_width))
    # Resize first, then pad to meet requested dimensions
    resized = cv2.resize(img, (resized_width, resized_height))
    r_height, r_width = resized.shape[:2]
    paddings = [
        [p_height, height - r_height - p_height],
        [p_width, width - r_width - p_width],
        [0, 0],
    ]
    padded = np.pad(resized, paddings, mode="constant")
    return padded


class CaribbeanDataset(Iterator):
    """
    """

    augmentation_params = {"buffer_jitter": (-2.5, 2.5), "angle_jitter": (-15, 15)}

    def __init__(
        self,
        dataframes,
        zone_to_image,
        id_col=None,
        label_col=None,
        batch_size=1,
        crop_buffer=0.5,
        augment=False,
        seed=0,
        image_preprocessing=None,
        label_preprocessing=None,
        img_shape=None,
        return_ids=False,
        return_labels=True,
        n_classes=-1,
        smooth_factor=0,
        smooth_ids_path=None,
        *args,
        **kwargs,
    ):
        """
        Creates a Keras dataset iterator from GeoDataFrames and raster images.
        Args:
            dataframes: list of GeoDataFrames which may have different crs.
            zone_to_image: dictionary that maps the DataFrame zone into a raster path.
            crop_buffer: buffer object crop a little bit (float wtih meter units).
            id_col (optional): specifies which DataFrame column use as example id.
            label_col (optional): specifies which DataFrame column use as label.
        """
        # Create offsets from dataframes lengths
        self._offsets = np.cumsum([0] + [len(df) for df in dataframes[:-1]])
        # Store dataframes with index continuity from one another
        self._dataframes = []
        for offset, df in zip(self._offsets, dataframes):
            df = df.reset_index(drop=True)
            df.index = df.index + offset
            self._dataframes.append(df)
        # Dictionary that maps zone into raster paths
        self._zone_to_image = zone_to_image
        self._num_examples = sum([len(df) for df in self._dataframes])
        self._label_col = label_col
        self._id_col = id_col
        self._crop_buffer = crop_buffer
        self._augment = augment
        self._img_shape = img_shape
        self._n_classes = n_classes
        self._return_ids = return_ids
        self._return_labels = return_labels
        self._smooth_factor = smooth_factor
        # Raster file handlers
        self._raster_handlers = {k: rasterio.open(v) for k, v in zone_to_image.items()}
        # Preprocessing image function
        self.image_preprocessing = image_preprocessing
        self.label_preprocessing = label_preprocessing
        self.smooth_ids = None
        if smooth_ids_path is not None:
            self._smooth_ids = pd.read_csv(outpath, names=["ids"])[
                "ids"
            ].values.tolist()
            logging.info(f"Read {len(self._smooth_ids)} smoothing ids.")
        # Init ImageDataGenerator
        super().__init__(
            n=self._num_examples,
            batch_size=batch_size,
            shuffle=augment,
            seed=seed,
            *args,
            **kwargs,
        )

    def _get_dataframe(self, idx):
        dataframe_idx = -1
        for j, offset in enumerate(self._offsets[1:]):
            if idx < offset:
                dataframe_idx = j
                break
        return self._dataframes[dataframe_idx]

    def _pad_and_stack(self, arrays, stack_axis=0):
        """Pad and stack a list of arrays."""
        shape_list = [arr.shape for arr in arrays]
        shapes = np.stack(shape_list, -1)
        max_shape = np.max(shapes, axis=-1)
        output_array = []
        for shape, arr in zip(shape_list, arrays):
            # Pad tensor
            paddings = np.stack(
                [np.zeros_like(shape), max_shape - shape], axis=-1
            ).tolist()
            if len(paddings) > 0:
                output_array += [np.pad(arr, paddings, mode="constant")]
            else:
                output_array += [arr]
        return np.stack(output_array, stack_axis)

    def _get_batches_of_transformed_samples(self, idxs):
        """
        Get batch of samples.
        """
        logging.debug(f"Retrieving samples for: {idxs}")
        # Create list of tuples [(id, img, label), (id, img, label)...]
        list_of_tuples = [self._get_sample(idx) for idx in idxs]
        # Create batch by transposing list of tuples into list of lists
        example_ids, batch_X, batch_Y = [
            list(tuples) for tuples in zip(*list_of_tuples)
        ]
        # Stack lists into numpy arrays
        if example_ids[0] is not None:
            example_ids = np.asarray(example_ids)
        else:
            example_ids = None
        if batch_Y[0] is not None:
            batch_Y = keras.utils.to_categorical(batch_Y, num_classes=self._n_classes)
            # label smoothing
            if self._smooth_factor:
                smoothed_batch_Y = []
                for example_id, Y in zip(example_ids, batch_Y):
                    if example_id in self._smooth_ids:
                        smooth_factor = np.random.uniform(0, self._smooth_factor)
                        smoothed_Y = smooth_labels(batch_Y, smooth_factor)
                        smoothed_batch_Y.append(smoothed_Y)
                    else:
                        smoothed_batch_Y.append(Y)
                batch_Y = np.asarray(smoothed_batch_Y)
        else:
            batch_Y = None
        # Pad and stack images
        # batch_X = self._pad_and_stack(batch_X)
        batch_X = np.stack(batch_X, axis=0)
        # return outputs according to _return_ids and _return_labels
        batch = (batch_X,)
        if self._return_labels:
            batch = batch + (batch_Y,)
        if self._return_ids:
            batch = (example_ids,) + batch
        return batch

    def _get_sample(self, idx):
        """
        Wrapper around ___get_sample__.
        """
        try:
            return self.___get_sample__(idx)
        except:
            logging.error(f"Error processing sample {idx}")
            logging.error(self._get_dataframe(idx).loc[idx, :])
            raise

    @retry(reraise=True, stop=stop_after_attempt(4), wait=wait_random(min=0.1, max=1))
    def _extract_from_raster(self, zone, geom):
        """
        Extract crop from raster.
        """
        raster = self._raster_handlers[zone]
        return rasterio.mask.mask(dataset=raster, shapes=[geom], crop=True)

    def ___get_sample__(self, idx):
        """
        """
        # Retrieve dataframe row for idx
        row = self._get_dataframe(idx).loc[idx, :]
        # Get shapely geometry from row
        geom = row.geometry
        if self._crop_buffer > 0:
            geom = geom.buffer(self._crop_buffer)
        if self._augment:
            aug_buffer = np.random.uniform(*self.augmentation_params["buffer_jitter"])
            aug_geom = geom.buffer(aug_buffer)
            geom_area = geom.area
            aug_geom_area = aug_geom.area
            # Used augmented geometry only when the area changes less than 10%
            # and the augmented area is greater than 2m2
            if abs(aug_geom_area - geom_area) / geom_area < 0.1 and aug_geom_area > 2.0:
                geom = aug_geom
        # Load image data
        out_img, out_transform = self._extract_from_raster(row["zone"], geom)
        out_img = np.transpose(out_img, [1, 2, 0])
        mask = out_img[..., -1]
        # Crop and fix its rotation
        result = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # hack for different opencv versions
        contours, hierarchy = result if len(result) == 2 else result[1:3]
        center, _, angle = cv2.minAreaRect(contours[0])
        if self._augment:
            angle_jitter = np.random.uniform(*self.augmentation_params["angle_jitter"])
            angle += angle_jitter
        rot = cv2.getRotationMatrix2D(center, angle - 90, 1)
        new_shape = tuple([max(out_img.shape[:2]) * 2] * 2)
        img = cv2.warpAffine(out_img[..., :3], rot, new_shape)
        mask = cv2.warpAffine(mask, rot, new_shape)
        # Crop image
        thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
        # Find contour and sort by contour area
        result = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts, hierarchy = result if len(result) == 2 else result[1:3]
        x, y, w, h = cv2.boundingRect(cnts[0])
        crop_img = img[y : y + h, x : x + w].copy()
        h, w, _ = crop_img.shape
        if h > w:
            crop_img = np.transpose(crop_img, [1, 0, 2])
        # Retrieve label
        if self._label_col is None:
            label = None
        else:
            label = row[self._label_col]
        # Retrieve id
        if self._id_col is None:
            example_id = None
        else:
            example_id = row[self._id_col]
        # Apply preprocessing function
        if self.image_preprocessing:
            crop_img = self.image_preprocessing(crop_img)
        if self._label_col is not None and self.label_preprocessing:
            label = self.label_preprocessing(label)
        return example_id, crop_img, label


def _read_geodataframe(path):
    df = pd.read_pickle(path).reset_index(drop=True)
    df = gpd.GeoDataFrame(df)
    df.crs = df.at[0, "crs"]
    df = df.drop(columns="crs")
    return df


def get_dataset_generator(
    imgs_pattern,
    dataframes_pattern,
    batch_size,
    img_dims,
    smooth_factor=0.0,
    smooth_ids_path=None,
    augment=False,
    return_ids=False,
    return_labels=True,
):
    """
    Function wrapper around CaribbeanDataset.
    Args:
        imgs_pattern: string that defines where to find image data.
        dataframes_pattern: string that defines where to find DataFrame data.
        batch_size: int that defines the batch size.
        img_dims: tuple or list that determines image dimensions.
    Example:
        dataframes_pattern = "../../data/processed/train/*.pkl"
        imgs_pattern = "../../data/raw/**/**/*.tif"
        gen = get_dataset_generator(imgs_pattern, dataframes_pattern, 1, (300, 300))
        example_id, X, y = next(gen) # X.shape==(1, 300, 300, 3), y.shape=[1]
    """
    # Get geo tif paths
    img_paths = glob(imgs_pattern)
    if not len(img_paths):
        raise ValueError(f"No image data found for pattern: {imgs_pattern}")
    # Create zone to image mapper
    zone_to_image = {p.split("/")[-2]: p for p in img_paths}
    logging.info(f"Mapping zones to images: {str(zone_to_image)}")
    # Read GeoDataFrames
    dataframe_paths = glob(dataframes_pattern)
    if not len(dataframe_paths):
        raise ValueError(f"No image data found for pattern: {dataframe_paths}")
    dataframes = [_read_geodataframe(path) for path in dataframe_paths]

    dataset = CaribbeanDataset(
        dataframes,
        zone_to_image,
        label_col="roof_material" if return_labels else None,
        id_col="id" if return_ids else None,
        batch_size=batch_size,
        augment=augment,
        return_ids=return_ids,
        return_labels=return_labels,
        n_classes=len(LABELMAP),
        smooth_factor=smooth_factor,
        smooth_ids_path=smooth_ids_path,
        image_preprocessing=lambda x: resize_with_pad(x, *img_dims),
        label_preprocessing=lambda x: LABELMAP[x],
        crop_buffer=0.5,
        img_shape=img_dims,
    )
    return dataset

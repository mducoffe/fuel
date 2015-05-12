from contextlib import closing

import numpy
from PIL import Image


def pil_imread_rgb(f):
    """Read an image with PIL, convert to RGB if necessary.

    Parameters
    ----------
    f : str or file-like object
        Filename or file object from which to read image data.

    Returns
    -------
    image : ndarray, 3-dimensional
        RGB image data as a NumPy array with shape `(rows, cols, 3)`.

    """
    with closing(Image.open(f)) as f:
        return numpy.array(f.convert('RGB'))


def square_crop(image, dim):
    """Crop an image to the central square after resizing it.

    Parameters
    ----------
    image : ndarray, 3-dimensional
        An image represented as a 3D ndarray, with 3 color
        channels represented as the third axis.
    dim : int, optional
        The length of the shorter side after resizing, and the
        length of both sides after cropping. Default is 256.

    Returns
    -------
    cropped : ndarray, 3-dimensional, shape `(dim, dim, 3)`
        The image resized such that the shorter side is length
        `dim`, with the longer side cropped to the central
        `dim` pixels.

    Notes
    -----
    This reproduces the preprocessing technique employed in [Kriz]_.

    .. [Kriz] A. Krizhevsky, I. Sutskever and G.E. Hinton (2012).
       "ImageNet Classification with Deep Convolutional Neural Networks."
       *Advances in Neural Information Processing Systems 25* (NIPS 2012).

    """
    if image.ndim != 3 and image.shape[2] != 3:
        raise ValueError("expected a 3-dimensional ndarray with last axis 3")
    if image.shape[0] > image.shape[1]:
        new_size = int(round(image.shape[0] / image.shape[1] * dim)), dim
        pad = (new_size[0] - dim) // 2
        slices = (slice(pad, pad + dim), slice(None))
    else:
        new_size = dim, int(round(image.shape[1] / image.shape[0] * dim))
        pad = (new_size[1] - dim) // 2
        slices = (slice(None), slice(pad, pad + dim))
    with closing(Image.fromarray(image, mode='RGB')) as pil_image:
        # PIL uses width x height, e.g. cols x rows, hence new_size backwards.
        resized = numpy.array(pil_image.resize(new_size[::-1], Image.BICUBIC))
    out = resized[slices]
    return out


def reshape_hwc_to_bchw(image):
    """Reshape an image to `(1, channels, image_height, image_width)`.

    Parameters
    ----------
    image : ndarray, shape `(image_height, image_width, channels)`
        The image to be resized.

    Returns
    -------
    ndarray, shape `(1, channels, image_height, image_width)`
        The same image data with the order of axes swapped, and a
        singleton leading axis added, making it convenient to
        pass a sequence of these to :func:`numpy.concatenate`.

    Notes
    -----
    `(batch, channel, height, width)` is the standard format for Theano's
    and cuDNN's convolution operations, so it makes most sense to store
    data in that layout. Adding the singleton batch size makes
    concatenating convenient.

    """
    return image.transpose(2, 0, 1)[numpy.newaxis]

from __future__ import division
from contextlib import closing
from functools import partial
import gzip
import io
import itertools
import os
import logging
import os.path
import sys
import tarfile

import h5py
import numpy
from picklable_itertools.extras import equizip
from scipy.io.matlab import loadmat
from six.moves import zip, xrange
from toolz.itertoolz import partition_all
import zmq

from fuel.datasets import H5PYDataset
from fuel.server import send_arrays, recv_arrays
from fuel.utils.formats import tar_open
from fuel.utils.image import pil_imread_rgb, square_crop, reshape_hwc_to_bchw
from fuel.utils.logging import (log_keys_values, SubprocessFailure,
                                ProgressBarHandler, zmq_log_and_monitor,
                                configure_zmq_process_logger)
from fuel.utils.zmq import (uninterruptible,
                            DivideAndConquerVentilator,
                            DivideAndConquerWorker,
                            DivideAndConquerSink,
                            LocalhostDivideAndConquerManager)

log = logging.getLogger(__name__)

DEVKIT_ARCHIVE = 'ILSVRC2010_devkit-1.0.tar.gz'
DEVKIT_META_PATH = 'devkit-1.0/data/meta.mat'
DEVKIT_VALID_GROUNDTRUTH_PATH = ('devkit-1.0/data/'
                                 'ILSVRC2010_validation_ground_truth.txt')
PATCH_IMAGES_TAR = 'patch_images.tar'
TEST_GROUNDTRUTH = 'ILSVRC2010_test_ground_truth.txt'
TRAIN_IMAGES_TAR = 'ILSVRC2010_images_train.tar'
VALID_IMAGES_TAR = 'ILSVRC2010_images_val.tar'
TEST_IMAGES_TAR = 'ILSVRC2010_images_test.tar'
IMAGE_TARS = TRAIN_IMAGES_TAR, VALID_IMAGES_TAR, TEST_IMAGES_TAR
ALL_FILES = IMAGE_TARS + (TEST_GROUNDTRUTH, DEVKIT_ARCHIVE, PATCH_IMAGES_TAR)


def ilsvrc2010(input_directory, save_path, image_dim=256,
               shuffle_train_set=True, shuffle_seed=(2015, 4, 1),
               num_workers=6, worker_batch_size=1024,
               output_filename='ilsvrc2010.hdf5'):
    """Converter for data from the ImageNet Large Scale Visual Recognition
    Challenge (ILSVRC) 2010 competition.

    Source files for this dataset can be obtained by registering at
    [ILSVRC2010WEB].

    Parameters
    ----------
    input_directory : str
        Path from which to read raw data files.
    save_path : str
        Path to which to save the HDF5 file.
    image_dim : int, optional
        The number of rows and columns to which images are normalized
        (default 256).
    shuffle_train_set : bool, optional
        If `True` (default), shuffle the training set within the HDF5 file,
        so that a sequential read through the training set is shuffled
        by default.
    shuffle_seed : int or sequence, optional
        Seed for a `numpy.random.RandomState` used to shuffle the training
        set order.
    num_workers : int, optional
        The number of worker processes to deploy.
    worker_batch_size : int, optional
        The number of images the workers should send to the sink at a
        time.
    output_filename : str, optional
        The output filename for the HDF5 file. Default: 'ilsvrc2010.hdf5'.

    .. [ILSVRC2010WEB] http://image-net.org/challenges/LSVRC/2010/index

    """
    debug = partial(partial(log_keys_values, process_type='MAIN'), log)

    # Read what's necessary from the development kit.
    devkit_path = os.path.join(input_directory, DEVKIT_ARCHIVE)
    synsets, cost_matrix, raw_valid_groundtruth = read_devkit(devkit_path)

    # Mapping to take WordNet IDs to our internal 0-999 encoding.
    wnid_map = dict(zip((s.decode('utf8') for s in synsets['WNID']),
                        xrange(1000)))

    train, valid, test, patch = [os.path.join(input_directory, fn)
                                 for fn in IMAGE_TARS + (PATCH_IMAGES_TAR,)]

    # Raw test data groundtruth, ILSVRC2010 IDs.
    raw_test_groundtruth = numpy.loadtxt(
        os.path.join(input_directory, TEST_GROUNDTRUTH),
        dtype=numpy.int16)

    # Ascertain the number of filenames to prepare appropriate sized
    # arrays.
    n_train = int(synsets['num_train_images'].sum())
    n_valid, n_test = len(raw_valid_groundtruth), len(raw_test_groundtruth)
    n_total = n_train + n_valid + n_test
    log.info('Training set: {} images'.format(n_train))
    log.info('Validation set: {} images'.format(n_valid))
    log.info('Test set: {} images'.format(n_test))
    log.info('Total (train/valid/test): {} images'.format(n_total))
    width = height = image_dim
    channels = 3
    with h5py.File(os.path.join(save_path, output_filename), 'w-') as f:
        log.info('Creating HDF5 datasets...')
        splits = {'train': (0, n_train),
                  'valid': (n_train, n_train + n_valid),
                  'test': (n_train + n_valid, n_total)}
        f.attrs['splits'] = H5PYDataset.create_split_array({
            'features': splits,
            'targets': splits,
            'filenames': splits
        })
        f.create_dataset('features', shape=(n_total, channels,
                                            height, width),
                         dtype=numpy.uint8)
        f.create_dataset('targets', shape=(n_total,),
                         dtype=numpy.int16)
        f.create_dataset('filenames', shape=(n_total,),
                         dtype='S32')
        log.info('Processing training set...')
        debug('STARTED_SET', which_set='train',
              total_images_in_set=n_train)
        process_train_set(f, train, patch, synsets['num_train_images'],
                          wnid_map, image_dim, num_workers,
                          worker_batch_size)
        debug('FINISHED_SET', which_set='train')
        ilsvrc_id_to_zero_based = dict(zip(synsets['ILSVRC2010_ID'],
                                       xrange(len(synsets))))
        valid_groundtruth = [ilsvrc_id_to_zero_based[id_]
                             for id_ in raw_valid_groundtruth]
        log.info('Processing validation set...')
        debug('STARTED_SET', which_set='valid',
              total_images_in_set=n_valid)
        for num_completed in process_other_set(f, valid, patch,
                                               valid_groundtruth,
                                               'valid', worker_batch_size,
                                               image_dim, n_train):
            debug('WRITTEN', which_set='valid',
                  num_images_written_so_far=num_completed)
        debug('FINISHED_SET', which_set='valid')
        test_groundtruth = [ilsvrc_id_to_zero_based[id_]
                            for id_ in raw_test_groundtruth]
        log.info('Processing test set...')
        debug('STARTED_SET', which_set='test',
              total_images_in_set=n_test)
        for num_completed in process_other_set(f, test, patch,
                                               test_groundtruth,
                                               'test', worker_batch_size,
                                               image_dim, n_train + n_valid):
            debug('WRITTEN', which_set='test',
                  num_images_written_so_far=num_completed)
        debug('FINISHED_SET', which_set='test')


class TrainSetProcessingManager(LocalhostDivideAndConquerManager):
    def __init__(self, logging_port, *args, **kwargs):
        super(TrainSetProcessingManager, self).__init__(*args, **kwargs)
        self.logging_port = logging_port

    def wait(self):
        terminate = False
        context = zmq.Context()
        try:
            zmq_log_and_monitor(self.logger, context,
                                processes=[self.ventilator_process,
                                           self.sink_process],
                                logging_port=self.logging_port,
                                failure_threshold=logging.ERROR)
        except KeyboardInterrupt:
            terminate = True
            log.info('Keyboard interrupt received.')
        except SubprocessFailure:
            terminate = True
            log.info('One or more substituent processes failed.')
        except Exception:
            terminate = True
        finally:
            log.info('Shutting down child processes...')
            self.cleanup()
            log.info('Killed child processes.')
            context.destroy()
            if terminate:
                sys.exit(1)


def process_train_set(hdf5_file, train_archive, patch_archive,
                      train_images_per_class, wnid_map, image_dim,
                      num_workers, worker_batch_size):
    """Process the ILSVRC2010 training set.

    Parameters
    ----------
    hdf5_file : :class:`h5py.File` instance
        HDF5 file handle to which to write. Assumes `features`, `targets`
        and `filenames` already exist and have first dimension larger than
        `sum(images_per_class)`.
    train_archive :  str or file-like object
        Filename or file handle for the TAR archive of training images.
    patch_archive :  str or file-like object
        Filename or file handle for the TAR archive of patch images.
    train_images_per_class : sequence
        A list of integers, where each element is the number of training
        set images for the corresponding class index.
    wnid_map : dict
        A dictionary mapping WordNet IDs to class indices.
    image_dim : int
        The width and height of the desired images after resizing and
        central cropping.
    num_workers : int
        The number of worker processes to spawn, in addition to a
        source and sink process.
    worker_batch_size : int
        The number of images each worker should send over the socket
        to the sink at a time.

    """
    ventilator = TrainSetVentilator(train_archive, logging_port=5559)
    workers = [TrainSetWorker(patch_archive, wnid_map, train_images_per_class,
                              image_dim, worker_batch_size, logging_port=5559)
               for _ in xrange(num_workers)]
    # TODO: additional arguments: flush_frequency, shuffle_seed
    sink = TrainSetSink(hdf5_file, train_images_per_class, logging_port=5559)
    manager = TrainSetProcessingManager(ventilator=ventilator, sink=sink,
                                        workers=workers, ventilator_port=5556,
                                        sink_port=5558, logging_port=5559)
    manager.launch()
    manager.wait()


def process_other_set(hdf5_file, archive, patch_archive, groundtruth,
                      which_set, worker_batch_size, image_dim, offset):
    """Process and convert either the validation set or the test set.

    Parameters
    ----------
    hdf5_file : :class:`h5py.File` instance
        HDF5 file handle to which to write. Assumes `features`, `targets`
        and `filenames` already exist and are at least as long as
        `offset` plus the number of files in `archive`.
    archive : str or file-like object
        The path or file-handle containing the TAR file of images to be
        processed.
    patch_archive : str or file-like object
        The path or file-handle containing the TAR file of patch
        images (see :func:`extract_patch_images`).
    groundtruth : ndarray, 1-dimensional
        Integer targets, with the same length as the number of images
    which_set : str
        One of 'valid' or 'test', used to extract the right patch images.
    worker_batch_size : int
        The number of examples/targets to write at a time.
    image_dim : int
        The width and height of the desired images after resizing and
        central cropping.
    offset : int
        The offset in the `features` and `targets` arrays at which to
        begin writing rows.

    Yields
    ------
    int
        A stream of integers. Each represents the number of examples
        processed so far, in increments of `worker_batch_size`.

    """
    features = hdf5_file['features']
    targets = hdf5_file['targets']
    filenames = hdf5_file['filenames']
    patch_images = extract_patch_images(patch_archive, which_set)
    with tar_open(archive) as tar:
        start = offset
        work_iter = cropped_resized_images_from_tar(tar, patch_images,
                                                    image_dim, groundtruth)
        for tuples_batch in partition_all(worker_batch_size, work_iter):
            images, labels, files = zip(*tuples_batch)
            this_chunk = len(images)
            features[start:start + this_chunk] = numpy.concatenate(images)
            targets[start:start + this_chunk] = labels
            filenames[start:start + this_chunk] = [f.encode('ascii')
                                                   for f in files]
            start += this_chunk
            yield start - offset


class HasZMQProcessLogger(object):
    """Mixin that adds logic for seting up a ZMQ logging handler."""
    def initialize_sockets(self, *args, **kwargs):
        super(HasZMQProcessLogger, self).initialize_sockets(*args, **kwargs)
        configure_zmq_process_logger(self.logger, self.context,
                                     self.logging_port)


class TrainSetVentilator(HasZMQProcessLogger, DivideAndConquerVentilator):
    """Serves per-class TAR files to workers.

    Parameters
    ----------
    train_archive : str or file-like object
        The TAR file containing the training set.
    logging_port : int
        The port on localhost on which to open a `PUSH` socket
        for sending :class:`logging.LogRecord`s.

    """
    def __init__(self, train_archive, logging_port, **kwargs):
        super(TrainSetVentilator, self).__init__(**kwargs)
        self.train_archive = train_archive
        self.logging_port = logging_port
        self.number = 0

    def produce(self):
        with tar_open(self.train_archive) as tar:
            for i, info in enumerate(tar):
                with closing(tar.extractfile(info.name)) as inner:
                    yield (i, info.name, inner.read())

    def send(self, socket, batch):
        i, name, data = batch
        self.number += 1
        self.debug('SENDING_TAR', tar_filename=name, number=self.number)
        uninterruptible(socket.send_pyobj, (i, name), zmq.SNDMORE)
        uninterruptible(socket.send, data)
        self.debug('SENT_TAR', tar_filename=name, number=self.number)


class TrainSetWorker(HasZMQProcessLogger, DivideAndConquerWorker):
    """Receives per-class TARs; extracts, crops, resizes and sends JPEGs.

    Parameters
    ----------
    patch_archive :  str or file-like object
        Filename or file handle for the TAR archive of patch images.
    wnid_map : dict
        A dictionary mapping WordNet IDs to class indices.
    images_per_class : sequence of int
        A sequence containing the number of images in each class.
        The sequence contains as many elements as there are classes.
    image_dim : int
        The width and height of the desired images after resizing and
        central cropping.
    batch_size : int, optional
        The number of images the workers should send to the sink at a
        time.
    logging_port : int, optional
        The port on localhost on which to open a `PUSH` socket
        for sending :class:`logging.LogRecord`s (default: 5559).

    """
    def __init__(self, patch_archive, wnid_map, images_per_class,
                 image_dim, batch_size, logging_port=5559, **kwargs):
        super(TrainSetWorker, self).__init__(**kwargs)
        self.patch_images = extract_patch_images(patch_archive, 'train')
        self.wnid_map = wnid_map
        self.images_per_class = images_per_class
        self.image_dim = image_dim
        self.batch_size = batch_size
        self.logging_port = logging_port

    def recv(self, socket):
        number, name = uninterruptible(socket.recv_pyobj)
        data = io.BytesIO(uninterruptible(socket.recv))
        return number, name, data

    def send(self, socket, results):
        label, images, filenames = results
        self.debug('SENDING_BATCH', tar_filename=self.current_tar_filename,
                   number=self.current_tar_number, num_images=len(images),
                   total_so_far=self.current_tar_images_processed, label=label)
        uninterruptible(socket.send_pyobj, label, zmq.SNDMORE)
        uninterruptible(send_arrays, socket, [images, filenames])
        self.debug('SENT_BATCH', tar_filename=self.current_tar_filename,
                   number=self.current_tar_number,
                   num_images=len(images), label=label,
                   total_so_far=(self.current_tar_images_processed +
                                 len(images)))

    def process(self, batch):
        number, name, tar_data = batch
        label = self.wnid_map[name.split('.')[0]]
        self.debug('RECEIVED_TAR', tar_filename=name, number=number,
                   label_id=label)
        with tarfile.open(fileobj=tar_data) as tar:
            self.debug('OPENED', tar_filename=name, number=number)
            self.current_tar_images_processed = 0
            self.current_tar_filename = name
            self.current_tar_number = number
            # Send images to sink in batches of at most worker_batch_size.
            combined = cropped_resized_images_from_tar(tar, self.patch_images,
                                                       self.image_dim)
            for tuples in partition_all(self.batch_size, combined):
                images, files, _ = zip(*tuples)
                yield (label, numpy.concatenate(images),
                       numpy.array(files, dtype='S32'))
                self.current_tar_images_processed += len(images)
        if self.current_tar_images_processed != self.images_per_class[label]:
            log.error('WORKER(%d): For class %s (%d), expected %d images but '
                      'only found %d', os.getpid(), name.split('.')[0], label,
                      self.images_per_class[label],
                      self.current_tar_images_processed)
        self.debug('FINISHED_TAR', tar_filename=name, number=number,
                   total=self.current_tar_images_processed, label=label)

    def handle_exception(self):
        log.error('%s(%d): Encountered error processing %s '
                  '(%d images processed successfully)',
                  self.process_type, os.getpid(), self.current_filename,
                  self.images_processed, exc_info=1)


class TrainSetSink(HasZMQProcessLogger, DivideAndConquerSink):
    """Writes incoming batches of processed images to a given HDF5 file.

    Parameters
    ----------
    hdf5_file : :class:`h5py.File` instance
        The HDF5 file with `features`, `targets` and `filenames`
        datasets already created within.
    images_per_class : sequence of int
        A sequence containing the number of images in each class.
        The sequence contains as many elements as there are classes.
    logging_port : int, optional
        The port on localhost on which to open a `PUSH` socket
        for sending :class:`logging.LogRecord`s (default: 5559).
    flush_frequency : int, optional
        How often, in number of batches, to call the `flush` method of
        `hdf5_file` (default: 256).
    shuffle_seed : int or sequence, optional
        The seed to use for the random number generator that determines
        the training set shuffling order.

    """
    def __init__(self, hdf5_file, images_per_class,
                 flush_frequency=256, logging_port=5559,
                 shuffle_seed=(2015, 4, 9), **kwargs):
        super(TrainSetSink, self).__init__(**kwargs)
        self.hdf5_file = hdf5_file
        self.flush_frequency = flush_frequency
        self.logging_port = logging_port
        self.shuffle_seed = shuffle_seed
        rng = numpy.random.RandomState(shuffle_seed)
        self.num_images_expected = sum(images_per_class)
        order = rng.permutation(self.num_images_expected)
        class_permutations = permutation_by_class(order, images_per_class)
        self.class_orders = [iter(o) for o in class_permutations]
        self.batches_received = 0
        self.num_images_written = 0
        self.images_sum = self._images_sq_sum = None

    def done(self):
        return self.num_images_written == self.num_images_expected

    def recv(self, socket):
        self.debug('RECEIVING_BATCH')
        label = uninterruptible(socket.recv_pyobj)
        images, filenames = uninterruptible(recv_arrays, socket)
        self.batches_received += 1
        self.debug('RECEIVED_BATCH', label=label,
                   num_images=images.shape[0], batch=self.batches_received)
        return label, images, filenames

    def process(self, batch):
        label, images, files = batch

        # Delay creation of the sum arrays until we've got the first
        # batch so that we can size them correctly.
        if self.images_sum is None:
            self.images_sum = numpy.zeros_like(images[0], dtype=numpy.float64)
            self.images_sq_sum = numpy.zeros_like(self.images_sum)

        # Grab the next few indices for this label. We partition the
        # indices by class so that no matter which order we receive
        # batches in, the final order is deterministic (because the
        # images within a class always appear in a deterministic order,
        # i.e. the order they are read out of the TAR file).
        indices = sorted(itertools.islice(self.class_orders[label],
                                          images.shape[0]))
        self.hdf5_file['features'][indices] = images
        labels = label * numpy.ones(images.shape[0], dtype=numpy.int16)
        self.hdf5_file['targets'][indices] = labels
        self.hdf5_file['filenames'][indices] = files

        self.num_images_written += images.shape[0]

        self.debug('WRITTEN', label=label,
                   num_images=images.shape[0], batch=self.batches_received,
                   num_images_written_so_far=self.num_images_written)

        # Accumulate the sum and the sum of the square, for mean and
        # variance statistics.
        self.images_sum += images.sum(axis=0)
        self.images_sq_sum += (images.astype(numpy.uint64) ** 2).sum(axis=0)

        # Manually flush file to disk at regular intervals. Unsure whether
        # this is strictly necessary.
        if self.batches_received % self.flush_frequency == 0:
            self.debug('FLUSH', hdf5_filename=self.hdf5_file.filename)
            self.hdf5_file.flush()

    def finalize(self):
        train_mean = self.images_sum / self.num_images_expected
        train_std = numpy.sqrt(self.images_sq_sum / self.num_images_expected -
                               train_mean**2)
        self.hdf5_file.create_dataset('features_train_mean',
                                      shape=train_mean.shape,
                                      dtype=train_mean.dtype)
        self.hdf5_file.create_dataset('features_train_std',
                                      shape=train_std.shape,
                                      dtype=train_std.dtype)
        self.hdf5_file['features_train_mean'][...] = train_mean
        self.hdf5_file['features_train_std'][...] = train_std
        self.hdf5_file['features'].dims.create_scale(
            self.hdf5_file['features_train_mean'], 'train_mean')
        self.hdf5_file['features'].dims.create_scale(
            self.hdf5_file['features_train_std'], 'train_std')
        self.hdf5_file['features'].dims[0].attach_scale(
            self.hdf5_file['features_train_mean'])
        self.hdf5_file['features'].dims[0].attach_scale(
            self.hdf5_file['features_train_std'])


def load_image_from_tar_or_patch(tar, image_filename, patch_images):
    """Do everything necessary to process a image inside a TAR.

    Parameters
    ----------
    tar : `TarFile` instance
        The tar from which to read `image_filename`.
    image_filename : str
        Fully-qualified path inside of `tar` from which to read an
        image file.
    patch_images : dict
        A dictionary containing filenames (without path) of replacements
        to be substituted in place of the version of the same file found
        in `tar`. Values are in `(width, height, channels)` layout.

    Returns
    -------
    ndarray
        An ndarray of shape `(height, width, 3)` representing an RGB image
        drawn either from the TAR file or the patch dictionary.

    """
    image = patch_images.get(os.path.basename(image_filename), None)
    if image is None:
        try:
            image = pil_imread_rgb(tar.extractfile(image_filename))
        except (IOError, OSError):
            with gzip.GzipFile(fileobj=tar.extractfile(image_filename)) as gz:
                image = pil_imread_rgb(gz)
    return image


def cropped_resized_images_from_tar(tar, patch_images, image_dim,
                                    groundtruth=None):
    """Generator that yields `(filename, image, label)` tuples.

    Parameters
    ----------
    tar : TarFile instance
        Open `TarFile` instance from which to read images.
    patch_images : dict
        A dictionary mapping (base, without-path) filenames to images
        which should be substituted for that filename rather than reading
        it from the TAR file.
    images_dim : int
        The width and height of the returned resized-and-square-cropped
        images (see :func:`fuel.utils.image.square_crop`).
    groundtruth : iterable, optional
        An iterable containing one integer label for each image in
        `filenames` (or each regular image in `tar` if `filenames`
        is left unspecified). Assumed to be sorted by filename of
        every regular file in `tar`. If `None`, a value of `None`
        will be emitted for every label.

    Yields
    ------
    filename
        A string representing the (base, without path) filename.
    image
        An image drawn from either the TAR archive or the `patch_images`
        dictionary (see :func:`load_image_from_tar_or_patch`), cropped
        and resized to `(image_dim, image_dim)` (see
        :func:`fuel.utils.image.square_crop`).
    label
        An integer label for the given image, or `None` if no groundtruth
        was available.

    """
    filenames = sorted(info.name for info in tar if info.isfile())
    if groundtruth is None:
        groundtruth = itertools.repeat(None, times=len(filenames))
    images_gen = (load_image_from_tar_or_patch(tar, filename, patch_images)
                  for filename in filenames)
    crops_gen = (square_crop(image, image_dim) for image in images_gen)
    reshapes_gen = (reshape_hwc_to_bchw(image) for image in crops_gen)
    filenames_gen = (os.path.basename(f) for f in filenames)
    reshapes_gen, filenames_gen
    for tup in equizip(reshapes_gen, groundtruth, filenames_gen):
        yield tup


def permutation_by_class(order, images_per_class):
    """Take a permutation on the integers and divide it into chunks.

    Parameters
    ----------
    order : sequence
        A sequence containing a permutation of the integers from
        0 to `len(order) - 1`.
    images_per_class : sequence
        A sequence containing the number of images in each class.
        The sequence contains as many elements as there are classes.

    Returns
    -------
    list of lists
        Each element of the returned list contains a number of
        elements corresponding to the same position in `images_per_class`;
        the elements of the inner lists are drawn sequentially from
        `order`.

    """
    if len(order) != sum(images_per_class):
        raise ValueError("images_per_class should sum to the length of order")
    result = []
    for num in images_per_class:
        result, order = result + [order[:num]], order[num:]
    return result


def read_devkit(f):
    """Read relevant information from the development kit archive.

    Parameters
    ----------
    f : str or file-like object
        The filename or file-handle for the gzipped TAR archive
        containing the ILSVRC2010 development kit.

    Returns
    -------
    synsets : ndarray, 1-dimensional, compound dtype
        See :func:`read_metadata` for details.
    cost_matrix : ndarray, 2-dimensional, uint8
        See :func:`read_metadata` for details.
    raw_valid_groundtruth : ndarray, 1-dimensional, int16
        The labels for the ILSVRC2010 validation set,
        distributed with the development kit code.

    """
    with tar_open(f) as tar:
        # Metadata table containing class hierarchy, textual descriptions, etc.
        meta_mat = tar.extractfile(DEVKIT_META_PATH)
        synsets, cost_matrix = read_metadata(meta_mat)

        # Raw validation data groundtruth, ILSVRC2010 IDs. Confusingly
        # distributed inside the development kit archive.
        raw_valid_groundtruth = numpy.loadtxt(tar.extractfile(
            DEVKIT_VALID_GROUNDTRUTH_PATH), dtype=numpy.int16)
    return synsets, cost_matrix, raw_valid_groundtruth


def read_metadata(meta_mat):
    """Read ILSVRC2010 metadata from the distributed MAT file.

    Parameters
    ----------
    meta_mat : str or file-like object
        The filename or file-handle for `meta.mat` from the
        ILSVRC2010 development kit.

    Returns
    -------
    synsets : ndarray, 1-dimensional, compound dtype
        A table containing ILSVRC2010 metadata for the "synonym sets"
        or "synsets" that comprise the classes and superclasses,
        including the following fields:
         * `ILSVRC2010_ID`: the integer ID used in the original
           competition data.
         * `WNID`: A string identifier that uniquely identifies
           a synset in ImageNet and WordNet.
         * `wordnet_height`: The length of the longest path to
           a leaf nodein the FULL ImageNet/WordNet hierarchy
           (leaf nodes in the FULL ImageNet/WordNet hierarchy
           have `wordnet_height` 0).
         * `gloss`: A string representation of an English
           textual description of the concept represented by
           this synset.
         * `num_children`: The number of children in the hierarchy
           for this synset.
         * `words`: A string representation, comma separated,
           of different synoym words or phrases for the concept
           represented by this synset.
         * `children`: A vector of `ILSVRC2010_ID`s of children
           of this synset, padded with -1. Note that these refer
           to `ILSVRC2010_ID`s from the original data and *not*
           the zero-based index in the table.
         * `num_train_images`: The number of training images for
           this synset.
    cost_matrix : ndarray, 2-dimensional, uint8
        A 1000x1000 matrix containing the precomputed pairwise
        cost (based on distance in the hierarchy) for all
        low-level synsets (i.e. the thousand possible output
        classes with training data associated).

    """
    mat = loadmat(meta_mat, squeeze_me=True)
    synsets = mat['synsets']
    cost_matrix = mat['cost_matrix']
    new_dtype = numpy.dtype([
        ('ILSVRC2010_ID', numpy.int16),
        ('WNID', ('S', max(map(len, synsets['WNID'])))),
        ('wordnet_height', numpy.int8),
        ('gloss', ('S', max(map(len, synsets['gloss'])))),
        ('num_children', numpy.int8),
        ('words', ('S', max(map(len, synsets['words'])))),
        ('children', (numpy.int8, max(synsets['num_children']))),
        ('num_train_images', numpy.uint16)
    ])
    new_synsets = numpy.empty(synsets.shape, dtype=new_dtype)
    for attr in ['ILSVRC2010_ID', 'WNID', 'wordnet_height', 'gloss',
                 'num_children', 'words', 'num_train_images']:
        new_synsets[attr] = synsets[attr]
    children = [numpy.atleast_1d(ch) for ch in synsets['children']]
    padded_children = [
        numpy.concatenate((c,
                           -numpy.ones(new_dtype['children'].shape[0] - len(c),
                                       dtype=numpy.int16)))
        for c in children
    ]
    new_synsets['children'] = padded_children
    return new_synsets, cost_matrix


def extract_patch_images(f, which_set):
    """Extracts a dict of the "patch images" for ILSVRC2010.

    Parameters
    ----------
    f : str or file-like object
        The filename or file-handle to the patch images TAR file.
    which_set : str
        Which set of images to extract. One of 'train', 'valid', 'test'.

    Returns
    -------
    dict
        A dictionary contains a mapping of filenames (without path) to a
        NumPy array containing the replacement image.

    Notes
    -----
    Certain images in the distributed archives are blank, or display
    an "image not available" banner. A separate TAR file of
    "patch images" is distributed with the corrected versions of
    these. It is this archive that this function is intended to read.

    """
    if which_set not in ('train', 'valid', 'test'):
        raise ValueError('which_set must be one of train, valid, or test')
    which_set = 'val' if which_set == 'valid' else which_set
    patch_images = {}
    with tar_open(f) as tar:
        for info_obj in tar:
            if not info_obj.name.endswith('.JPEG'):
                continue
            # Pretty sure that '/' is used for tarfile regardless of
            # os.path.sep, but I officially don't care about Windows.
            tokens = info_obj.name.split('/')
            file_which_set = tokens[1]
            if file_which_set != which_set:
                continue
            filename = tokens[-1]
            image = pil_imread_rgb(tar.extractfile(info_obj.name))
            patch_images[filename] = image
    return patch_images


if __name__ == "__main__":
    formatter = logging.Formatter('%(asctime)s %(message)s')
    fh = logging.FileHandler('log.txt')
    log.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    log.handlers.clear()
    log.addHandler(fh)
    while log.root.handlers:
        log.root.handlers.pop()
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    log.addHandler(sh)
    log.addHandler(ProgressBarHandler('total_images_in_set',
                                      'num_images_written_so_far'))
    ilsvrc2010('.', '.')

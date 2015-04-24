import errno
import pickle
import time
from six.moves import range, cPickle
from numpy.testing import assert_raises, assert_equal
from zmq import ZMQError
import zmq

from fuel.iterator import DataIterator
from fuel.utils import do_not_pickle_attributes
from fuel.utils.zmq import uninterruptible
from fuel.utils.zmq import (DivideAndConquerVentilator, DivideAndConquerSink,
                            DivideAndConquerWorker,
                            LocalhostDivideAndConquerManager)


@do_not_pickle_attributes("non_picklable", "bulky_attr")
class DummyClass(object):
    def __init__(self):
        self.load()

    def load(self):
        self.bulky_attr = list(range(100))
        self.non_picklable = lambda x: x


class FaultyClass(object):
    pass


@do_not_pickle_attributes("iterator")
class UnpicklableClass(object):
    def __init__(self):
        self.load()

    def load(self):
        self.iterator = DataIterator(None)


@do_not_pickle_attributes("attribute")
class NonLoadingClass(object):
    def load(self):
        pass


class TestDoNotPickleAttributes(object):
    def test_load(self):
        instance = cPickle.loads(cPickle.dumps(DummyClass()))
        assert_equal(instance.bulky_attr, list(range(100)))
        assert instance.non_picklable is not None

    def test_value_error_no_load_method(self):
        assert_raises(ValueError, do_not_pickle_attributes("x"), FaultyClass)

    def test_value_error_iterator(self):
        assert_raises(ValueError, cPickle.dumps, UnpicklableClass())

    def test_value_error_attribute_non_loaded(self):
        assert_raises(ValueError, getattr, NonLoadingClass(), 'attribute')


def test_uninterruptible():
    foo = []

    def interrupter(a, b):
        if len(foo) < 3:
            foo.append(0)
            raise zmq.ZMQError(errno=errno.EINTR)
        return (len(foo) + a) / b

    def noninterrupter():
        return -1

    assert uninterruptible(interrupter, 5,  2) == 4


class DummyVentilator(DivideAndConquerVentilator):
    def send(self, socket, number):
        socket.send_pyobj(number)

    def produce(self):
        for i in range(50):
            yield i


class DummyWorker(DivideAndConquerWorker):
    def recv(self, socket):
        return socket.recv_pyobj()

    def send(self, socket, number):
        socket.send_pyobj(number)

    def process(self, number):
        yield number ** 2


class DummySink(DivideAndConquerSink):
    def __init__(self, result_port, sync_port):
        self.result_port = result_port
        self.sync_port = sync_port
        self.messages_received = 0
        self.sum = 0

    def recv(self, socket):
        print('Receiving message', self.messages_received)
        received = socket.recv_pyobj()
        print('Received!', received)
        self.messages_received += 1
        print('incremented, sum =', self.sum)
        return received

    def done(self):
        return self.messages_received >= 50

    def setup_sockets(self, context, *args, **kwargs):
        super(DummySink, self).setup_sockets(context, *args, **kwargs)
        self.publisher = publisher = self.context.socket(zmq.PUB)
        # set SNDHWM, so we don't drop messages for slow subscribers
        publisher.sndhwm = 1100000
        publisher.bind('tcp://*:{}'.format(self.result_port))

    def process(self, number_squared):
        print('Received', number_squared, 'for processing')
        self.sum += number_squared
        print('self.sum is now', self.sum)

    def shutdown(self):
        print('SHUTTING DOWN!', self.sum)
        self._receiver.close()

        # Socket to receive signals
        syncservice = self.context.socket(zmq.REP)
        syncservice.bind('tcp://*:{}'.format(self.sync_port))
        # wait for synchronization request
        syncservice.recv()
        # send synchronization reply
        syncservice.send(b'')
        self.publisher.send_pyobj(self.sum)


def test_localhost_divide_and_conquer_manager():
    result_port = 59581
    sync_port = 59582
    ventilator_port = 59583
    sink_port = 59584
    manager = LocalhostDivideAndConquerManager(DummyVentilator(),
                                               DummySink(result_port,
                                                         sync_port),
                                               [DummyWorker(), DummyWorker()],
                                               ventilator_port, sink_port)
    manager.launch()
    context = zmq.Context()

    # First, connect our subscriber socket
    subscriber = context.socket(zmq.SUB)
    subscriber.connect('tcp://localhost:{}'.format(result_port))
    subscriber.setsockopt(zmq.SUBSCRIBE, b'')
    time.sleep(0.5)
    # Second, synchronize with publisher
    syncclient = context.socket(zmq.REQ)
    syncclient.connect('tcp://localhost:{}'.format(sync_port))
    # send a synchronization request
    syncclient.send(b'')
    # wait for synchronization reply
    syncclient.recv()
    print('Receiving message (in test)')
    result = subscriber.recv_pyobj()
    print("Received", result, '(in test)')
    manager.wait_for_sink()
    assert result == sum(i ** 2 for i in range(50))

if __name__ == "__main__":
    test_localhost_divide_and_conquer_manager()

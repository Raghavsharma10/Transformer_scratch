def iterate(self, max_iter=None):
        """
        Note: A ZMQStreamer does not activate its stream,
        but allows the zmq_worker to do that.

        Yields
        ------
        data : dict
            Data drawn from `streamer(max_iter)`.
        """
        context = zmq.Context()

        if six.PY2:
            warnings.warn('zmq_stream cannot preserve numpy array alignment '
                          'in Python 2', RuntimeWarning)

        try:
            socket = context.socket(zmq.PAIR)

            port = socket.bind_to_random_port('tcp://*',
                                              min_port=self.min_port,
                                              max_port=self.max_port,
                                              max_tries=self.max_tries)
            terminate = mp.Event()

            worker = mp.Process(target=SafeFunction(zmq_worker),
                                args=[port, self.streamer, terminate],
                                kwargs=dict(copy=self.copy,
                                            max_iter=max_iter))

            worker.daemon = True
            worker.start()

            # Yield from the queue as long as it's open
            while True:
                yield zmq_recv_data(socket)

        except StopIteration:
            pass

        except:
            # pylint: disable-msg=W0702
            six.reraise(*sys.exc_info())

        finally:
            terminate.set()
            worker.join(self.timeout)
            if worker.is_alive():
                worker.terminate()
            context.destroy()
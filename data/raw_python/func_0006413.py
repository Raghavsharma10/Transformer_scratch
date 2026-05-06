def readout(self, fifo, no_data_timeout=None):
        '''Readout thread continuously reading FIFO.

        Readout thread, which uses read_raw_data_from_fifo() and appends data to self._fifo_data_deque (collection.deque).
        '''
        logging.info('Starting readout thread for %s', fifo)
        time_last_data = time()
        time_wait = 0.0
        empty_reads = 0
        while not self.force_stop[fifo].wait(time_wait if time_wait >= 0.0 else 0.0):
            time_read = time()
            try:
                if no_data_timeout and time_last_data + no_data_timeout < get_float_time():
                    raise NoDataTimeout('Received no data for %0.1f second(s) from %s' % (no_data_timeout, fifo))
                raw_data = self.read_raw_data_from_fifo(fifo)
            except NoDataTimeout:
                no_data_timeout = None  # raise exception only once
                if self.errback:
                    self.errback(sys.exc_info())
                else:
                    raise
            except Exception:
                if self.errback:
                    self.errback(sys.exc_info())
                else:
                    raise
                if self.stop_readout.is_set():  # in case of a exception, break immediately
                    break
            else:
                n_data_words = raw_data.shape[0]
                if n_data_words > 0:
                    time_last_data = time()
                    empty_reads = 0
                    time_start_read, time_stop_read = self.update_timestamp(fifo)
                    status = 0
                    self._fifo_data_deque[fifo].append((raw_data, time_start_read, time_stop_read, status))
                    with self._fifo_conditions[fifo]:
                        self._fifo_conditions[fifo].notify_all()
                elif self.stop_readout.is_set():
                    if empty_reads == self._n_empty_reads:
                        break
                    else:
                        empty_reads += 1
            finally:
                # ensure that the readout interval does not depend on the processing time of the data
                # and stays more or less constant over time
                time_wait = self.readout_interval - (time() - time_read)
        self._fifo_data_deque[fifo].append(None)  # last item, None will stop worker
        with self._fifo_conditions[fifo]:
            self._fifo_conditions[fifo].notify_all()
        logging.info('Stopping readout thread for %s', fifo)
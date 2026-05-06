def writer(self, index, no_data_timeout=None):
        '''Writer thread continuously calling callback function for writing data when data becomes available.
        '''
        is_fe_data_header = logical_and(is_fe_word, is_data_header)
        logging.debug('Starting writer thread with index %d', index)
        self._data_conditions[index].acquire()
        time_last_data = time()
        time_write = time()
        converted_data_tuple_list = [None] * len(self.filter_func)  # callback function gets a list of lists of tuples
        while True:
            try:
                if no_data_timeout and time_last_data + no_data_timeout < time():
                    raise NoDataTimeout('Received no data for %0.1f second(s) for writer thread with index %d' % (no_data_timeout, index))
                converted_data_tuple = self._data_deque[index].popleft()
            except NoDataTimeout:  # no data timeout
                no_data_timeout = None  # raise exception only once
                if self.errback:
                    self.errback(sys.exc_info())
                else:
                    raise
            except IndexError:  # no data in queue
                self._data_conditions[index].wait(self.readout_interval)  # sleep a little bit, reducing CPU usage
            else:
                if converted_data_tuple is None:  # if None then write and exit
                    if self.callback and any(converted_data_tuple_list):
                        try:
                            self.callback(converted_data_tuple_list)
                        except Exception:
                            self.errback(sys.exc_info())
                    break
                else:
                    if no_data_timeout and np.any(is_fe_data_header(converted_data_tuple[0])):  # check for FEI4 data words
                        time_last_data = time()
                    if converted_data_tuple_list[index]:
                        converted_data_tuple_list[index].append(converted_data_tuple)
                    else:
                        converted_data_tuple_list[index] = [converted_data_tuple]  # adding iterable
                    if self.fill_buffer:
                        self._data_buffer[index].append(converted_data_tuple)
            # check if calling the callback function is about time
            if self.callback and any(converted_data_tuple_list) and ((self.write_interval and time() - time_write >= self.write_interval) or not self.write_interval):
                try:
                    self.callback(converted_data_tuple_list)  # callback function gets a list of lists of tuples
                except Exception:
                    self.errback(sys.exc_info())
                else:
                    converted_data_tuple_list = [None] * len(self.filter_func)
                    time_write = time()  # update last write timestamp
        self._data_conditions[index].release()
        logging.debug('Stopping writer thread with index %d', index)
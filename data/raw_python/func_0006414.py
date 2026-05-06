def worker(self, fifo):
        '''Worker thread continuously filtering and converting data when data becomes available.
        '''
        logging.debug('Starting worker thread for %s', fifo)
        self._fifo_conditions[fifo].acquire()
        while True:
            try:
                data_tuple = self._fifo_data_deque[fifo].popleft()
            except IndexError:
                self._fifo_conditions[fifo].wait(self.readout_interval)  # sleep a little bit, reducing CPU usage
            else:
                if data_tuple is None:  # if None then exit
                    break
                else:
                    for index, (filter_func, converter_func, fifo_select) in enumerate(izip(self.filter_func, self.converter_func, self.fifo_select)):
                        if fifo_select is None or fifo_select == fifo:
                            # filter and do the conversion
                            converted_data_tuple = convert_data_iterable((data_tuple,), filter_func=filter_func, converter_func=converter_func)[0]
                            n_data_words = converted_data_tuple[0].shape[0]
                            with self.data_words_per_second_lock:
                                self._words_per_read[index].append((n_data_words, converted_data_tuple[1], converted_data_tuple[2]))
                            self._data_deque[index].append(converted_data_tuple)
                            with self._data_conditions[index]:
                                self._data_conditions[index].notify_all()
        for index, fifo_select in enumerate(self.fifo_select):
            if fifo_select is None or fifo_select == fifo:
                self._data_deque[index].append(None)
                with self._data_conditions[index]:
                    self._data_conditions[index].notify_all()
        self._fifo_conditions[fifo].release()
        logging.debug('Stopping worker thread for %s', fifo)
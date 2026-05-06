def handle_data(self, data, new_file=False, flush=True):
        '''Handling of the data.

        Parameters
        ----------
        data : list, tuple
            Data tuple of the format (data (np.array), last_time (float), curr_time (float), status (int))
        '''
        for i, module_id in enumerate(self._selected_modules):
            if data[i] is None:
                continue
            self._raw_data_files[module_id].append(data_iterable=data[i], scan_parameters=self._scan_parameters[module_id]._asdict(), new_file=new_file, flush=flush)
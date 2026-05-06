def read_data(self):
        """
            Read data from the ndata file reference

            :param reference: the reference from which to read
            :return: a numpy array of the data; maybe None
        """
        with self.__lock:
            absolute_file_path = self.__file_path
            #logging.debug("READ data file %s", absolute_file_path)
            with open(absolute_file_path, "rb") as fp:
                local_files, dir_files, eocd = parse_zip(fp)
                return read_data(fp, local_files, dir_files, b"data.npy")
            return None
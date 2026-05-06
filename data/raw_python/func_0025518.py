def remove(self):
        """
            Remove the ndata file reference

            :param reference: the reference to remove
        """
        with self.__lock:
            absolute_file_path = self.__file_path
            #logging.debug("DELETE data file %s", absolute_file_path)
            if os.path.isfile(absolute_file_path):
                os.remove(absolute_file_path)
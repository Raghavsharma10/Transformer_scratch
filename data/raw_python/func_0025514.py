def write_data(self, data, file_datetime):
        """
            Write data to the ndata file specified by reference.

            :param data: the numpy array data to write
            :param file_datetime: the datetime for the file
        """
        with self.__lock:
            assert data is not None
            absolute_file_path = self.__file_path
            #logging.debug("WRITE data file %s for %s", absolute_file_path, key)
            make_directory_if_needed(os.path.dirname(absolute_file_path))
            properties = self.read_properties() if os.path.exists(absolute_file_path) else dict()
            write_zip(absolute_file_path, data, properties)
            # convert to utc time.
            tz_minutes = Utility.local_utcoffset_minutes(file_datetime)
            timestamp = calendar.timegm(file_datetime.timetuple()) - tz_minutes * 60
            os.utime(absolute_file_path, (time.time(), timestamp))
def write_properties(self, properties, file_datetime):
        """
            Write properties to the ndata file specified by reference.

            :param reference: the reference to which to write
            :param properties: the dict to write to the file
            :param file_datetime: the datetime for the file

            The properties param must not change during this method. Callers should
            take care to ensure this does not happen.
        """
        with self.__lock:
            absolute_file_path = self.__file_path
            #logging.debug("WRITE properties %s for %s", absolute_file_path, key)
            make_directory_if_needed(os.path.dirname(absolute_file_path))
            exists = os.path.exists(absolute_file_path)
            if exists:
                rewrite_zip(absolute_file_path, Utility.clean_dict(properties))
            else:
                write_zip(absolute_file_path, None, Utility.clean_dict(properties))
            # convert to utc time.
            tz_minutes = Utility.local_utcoffset_minutes(file_datetime)
            timestamp = calendar.timegm(file_datetime.timetuple()) - tz_minutes * 60
            os.utime(absolute_file_path, (time.time(), timestamp))
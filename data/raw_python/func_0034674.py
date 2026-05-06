def write_log(self, fs=None):
        """
        Write the results of the benchmark to a log file.
        :param fs: file-like object.
        """
        log = StringIO.StringIO()
        log.write(self.setup_src)

        # If the function is not bound, write the test score to the log
        if not self.is_class_method:
            time_avg = convert_time_units(self.time_average_seconds)
            log.write("\nAverage time: {0} \n".format(time_avg))

        if fs:
            with open(fs, 'w') as _f:
                _f.write(log.getvalue())
def force_log_type_flags(self, logType, stdoutFlag, fileFlag):
        """
        Force a logtype logging flags.

        :Parameters:
           #. logType (string): A defined logging type.
           #. stdoutFlag (None, boolean): The standard output logging flag.
              If None, logtype stdoutFlag forcing is released.
           #. fileFlag (None, boolean): The file logging flag.
              If None, logtype fileFlag forcing is released.
        """
        self.force_log_type_stdout_flag(logType, stdoutFlag)
        self.force_log_type_file_flag(logType, fileFlag)
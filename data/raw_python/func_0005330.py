def get_logfile_name(tag):
        """
        Creates a name for a log file that is meant to be used in a call to
        ``logging.FileHandler``. The log file name will incldue the path to the log directory given
        by the `p.LOG_DIR` constant. The format of the file name is: 'log_$HOST_$TAG.txt', where

        $HOST is the hostname part of the URL given by ``URL``, and $TAG is the value of the
        'tag' argument. The log directory will be created if need be.

        Args:
            tag: `str`. A tag name to add to at the end of the log file name for clarity on the
                log file's purpose.
        """
        if not os.path.exists(p.LOG_DIR):
            os.mkdir(p.LOG_DIR)
        filename = "log_" + p.HOST + "_" + tag + ".txt"
        filename = os.path.join(p.LOG_DIR, filename)
        return filename
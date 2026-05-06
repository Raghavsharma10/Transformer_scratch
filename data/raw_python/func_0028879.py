def _get_exception_log_path():
        """Return the normalized path for the connection log, raising an
        exception if it can not written to.

        :return: str

        """
        app = sys.argv[0].split('/')[-1]
        for exception_log in ['/var/log/%s.errors' % app,
                              '/var/tmp/%s.errors' % app,
                              '/tmp/%s.errors' % app]:
            if os.access(path.dirname(exception_log), os.W_OK):
                return exception_log
        return None
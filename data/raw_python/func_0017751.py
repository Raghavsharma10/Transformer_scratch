def _get_logger_file_handles(self):
        """
        Find the file handles used by our logger's handlers.
        """
        handles = []
        for handler in self.logger.handlers:
            # The following code works for logging's SysLogHandler,
            # StreamHandler, SocketHandler, and their subclasses.
            for attr in ['sock', 'socket', 'stream']:
                try:
                    handle = getattr(handler, attr)
                    if handle:
                        handles.append(handle)
                    break
                except AttributeError:
                    continue
        return handles
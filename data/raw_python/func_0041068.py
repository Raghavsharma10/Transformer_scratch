def pformat(self, traceback=False):
        """Pretty formats the failure object into a string."""
        buf = six.StringIO()
        if not self._exc_type_names:
            buf.write('Failure: %s' % (self._exception_str))
        else:
            buf.write('Failure: %s: %s' % (self._exc_type_names[0],
                                           self._exception_str))
        if traceback:
            if self._traceback_str is not None:
                traceback_str = self._traceback_str.rstrip()
            else:
                traceback_str = None
            if traceback_str:
                buf.write(os.linesep)
                buf.write(traceback_str)
            else:
                buf.write(os.linesep)
                buf.write('Traceback not available.')
        return buf.getvalue()
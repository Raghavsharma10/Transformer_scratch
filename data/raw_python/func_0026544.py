def log(self, *args, **kwargs):
        """Log a statement from this component"""

        func = inspect.currentframe().f_back.f_code
        # Dump the message + the name of this function to the log.

        if 'exc' in kwargs and kwargs['exc'] is True:
            exc_type, exc_obj, exc_tb = exc_info()
            line_no = exc_tb.tb_lineno
            # print('EXCEPTION DATA:', line_no, exc_type, exc_obj, exc_tb)
            args += traceback.extract_tb(exc_tb),
        else:
            line_no = func.co_firstlineno

        sourceloc = "[%.10s@%s:%i]" % (
            func.co_name,
            func.co_filename,
            line_no
        )
        hfoslog(sourceloc=sourceloc, emitter=self.uniquename, *args, **kwargs)
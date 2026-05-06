def formatException(self, exc_info, record=None):
        """Format exception output with CONF.logging_exception_prefix."""
        if not record:
            return logging.Formatter.formatException(self, exc_info)

        stringbuffer = cStringIO.StringIO()
        traceback.print_exception(exc_info[0], exc_info[1], exc_info[2],
                                  None, stringbuffer)
        lines = stringbuffer.getvalue().split('\n')
        stringbuffer.close()

        if CONF.logging_exception_prefix.find('%(asctime)') != -1:
            record.asctime = self.formatTime(record, self.datefmt)

        formatted_lines = []
        for line in lines:
            pl = CONF.logging_exception_prefix % record.__dict__
            fl = '%s%s' % (pl, line)
            formatted_lines.append(fl)
        return '\n'.join(formatted_lines)
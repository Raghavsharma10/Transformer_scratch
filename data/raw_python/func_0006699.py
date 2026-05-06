def format(self, record):
        """Uses contextstring if request_id is set, otherwise default."""
        # NOTE(sdague): default the fancier formating params
        # to an empty string so we don't throw an exception if
        # they get used
        for key in ('instance', 'color'):
            if key not in record.__dict__:
                record.__dict__[key] = ''

        if record.__dict__.get('request_id', None):
            self._fmt = CONF.logging_context_format_string
        else:
            self._fmt = CONF.logging_default_format_string

        if (record.levelno == logging.DEBUG and
                CONF.logging_debug_format_suffix):
            self._fmt += " " + CONF.logging_debug_format_suffix

        # Cache this on the record, Logger will respect our formated copy
        if record.exc_info:
            record.exc_text = self.formatException(record.exc_info, record)
        return logging.Formatter.format(self, record)
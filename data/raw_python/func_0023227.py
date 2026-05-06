def _vispy_emit_match_andor_record(self, record):
        """Log message emitter that optionally matches and/or records"""
        test = record.getMessage()
        match = self._vispy_match
        if (match is None or re.search(match, test) or
                re.search(match, _get_vispy_caller())):
            if self._vispy_emit_record:
                fmt_rec = self._vispy_formatter.format(record)
                self._vispy_emit_list.append(fmt_rec)
            if self._vispy_print_msg:
                return logging.StreamHandler.emit(self, record)
            else:
                return
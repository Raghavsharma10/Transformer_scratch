def emit(self, record):
        """
        Emit a record.

        If a formatter is specified, it is used to format the record.
        If exception information is present, it is formatted using
        traceback.print_exception and sent to Storm.
        """
        try:
            msg = self.format(record)
            level = _STORM_LOG_LEVELS.get(record.levelname.lower(), _STORM_LOG_INFO)
            self.serializer.send_message(
                {"command": "log", "msg": str(msg), "level": level}
            )
        except Exception:
            self.handleError(record)
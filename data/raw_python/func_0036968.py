def format(self, record):
        """
        Formats a given log record to include the timestamp, log level, thread
        ID and message.  Colorized if coloring is available.
        """
        if not self.is_tty:
            return super(CLIHandler, self).format(record)

        level_abbrev = record.levelname[0]

        time_and_level = color_string(
            color_for_level(record.levelno),
            "[%(asctime)s " + level_abbrev + "]"
        )
        thread = color_string(
            color_for_thread(record.thread),
            "[%(threadName)s]"
        )
        formatter = logging.Formatter(
            time_and_level + thread + " %(message)s", "%Y-%m-%d %H:%M:%S"
        )

        return formatter.format(record)
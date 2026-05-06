def get_default_logger():
        """Returns default driver logger.

        :return: logger instance
        :rtype: logging.Logger
        """
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            "[%(levelname)1.1s %(asctime)s %(name)s] %(message)s",
            "%y%m%d %H:%M:%S"))

        logger_name = "pydbal"
        if Connection._instance_count > 1:
            logger_name += ":" + str(Connection._instance_count)
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        return logger
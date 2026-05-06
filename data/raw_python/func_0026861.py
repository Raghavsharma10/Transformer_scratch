def _ready(self):
        """
        check if ther is som data is ready to get read.
        :return True if there is some date
        :rtype bool
        """
        # if DOUT pin is low, data is ready for reading
        _is_ready = GPIO.input(self._dout) == 0
        logging.debug("check data ready for reading: {result}".format(
            result="YES" if _is_ready is True else "NO"
        ))
        return _is_ready
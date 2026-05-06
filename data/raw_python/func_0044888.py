def publish_attrs(self, upcount=1):
        """
        Magic function which inject all attrs into the callers namespace
        :param upcount     int, how many stack levels we go up
        :return:
        """

        frame = inspect.currentframe()
        i = upcount
        while True:
            if frame.f_back is None:
                break
            frame = frame.f_back
            i -= 1
            if i == 0:
                break

        for k, v in self.__dict__.items():
            frame.f_globals[k] = v
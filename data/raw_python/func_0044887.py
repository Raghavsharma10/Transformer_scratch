def fetch_locals(self, upcount=1):
        """
        Magic function which fetches all variables from the callers namespace
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

        for k, v in frame.f_locals.items():
            self.__dict__[k] = v
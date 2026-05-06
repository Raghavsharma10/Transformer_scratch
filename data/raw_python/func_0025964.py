def put(self, name, start, end):
        """
        Stores a new target.
        :param name: the name.
        :param start: start time.
        :param end: end time.
        :return:
        """
        entry = self._uploadController.getEntry(name)
        if entry is not None:
            return None, 200 if self._targetController.storeFromWav(entry, start, end) else 500
        else:
            return None, 404
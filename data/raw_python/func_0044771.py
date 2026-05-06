def id(self):
        """
        :return: the ID attribute of the sentence
        :rtype: int

        """
        if self._id is None:
            self._id = int(self._element.get('id'))
        return self._id
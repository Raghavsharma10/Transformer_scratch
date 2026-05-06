def shadow_copy(self):
        """ Return a copy of the resource with same raw data

        :return: copy of the resource
        """
        ret = self.__class__()
        if not self._is_updated():
            # before copy, make sure source is updated.
            self.update()
        ret._parsed_resource = self._parsed_resource
        return ret
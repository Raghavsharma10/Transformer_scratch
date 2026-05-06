def dict(self):
        """A dict that holds key/values for all of the properties in the
        object.

        :return:

        """
        d = {p.key: getattr(self, p.key) for p in self.__mapper__.attrs
             if p.key not in ('contents', 'dataset')}

        d['modified_datetime'] = self.modified_datetime
        d['modified_ago'] = self.modified_ago

        return d
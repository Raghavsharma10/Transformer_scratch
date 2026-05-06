def slice_naive(self, key):
        """
        Naively slice each data object in the container by the object's index.

        Args:
            key: Int, slice, or list by which to extra "sub"-container

        Returns:
            sub: Sub container of the same format with a view of the data

        Warning:
            To ensure that a new container is created, use the copy method.

            .. code-block:: Python

                mycontainer[slice].copy()
        """
        kwargs = {'name': self.name, 'description': self.description, 'meta': self.meta}
        for name, data in self._data().items():
            k = name[1:] if name.startswith('_') else name
            kwargs[k] = data.slice_naive(key)
        return self.__class__(**kwargs)
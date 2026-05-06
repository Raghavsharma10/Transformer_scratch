def copy(self, name=None, description=None, meta=None):
        """
        Create a copy of the current object (may alter the container's name,
        description, and update the metadata if needed).
        """
        cls = self.__class__
        kwargs = self._rel(copy=True)
        kwargs.update(self._data(copy=True))
        if name is not None:
            kwargs['name'] = name
        if description is not None:
            kwargs['description'] = description
        if meta is not None:
            kwargs['meta'] = meta
        return cls(**kwargs)
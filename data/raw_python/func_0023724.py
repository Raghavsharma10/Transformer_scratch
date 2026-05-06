def _reset(self, **kwargs):
        """
        Reset after repopulating from API.
        """

        # there are some inconsistenciens in the API regarding these
        # note: this could be written in fancier ways, but this way is simpler

        if 'uuid' in kwargs:
            self.uuid = kwargs['uuid']
        elif 'storage' in kwargs:  # let's never use storage.storage internally
            self.uuid = kwargs['storage']

        if 'title' in kwargs:
            self.title = kwargs['title']
        elif 'storage_title' in kwargs:
            self.title = kwargs['storage_title']

        if 'size' in kwargs:
            self.size = kwargs['size']
        elif 'storage_size' in kwargs:
            self.size = kwargs['storage_size']

        # send the rest to super._reset

        filtered_kwargs = dict(
            (key, val)
            for key, val in kwargs.items()
            if key not in ['uuid', 'storage', 'title', 'storage_title', 'size', 'storage_size']
        )
        super(Storage, self)._reset(**filtered_kwargs)
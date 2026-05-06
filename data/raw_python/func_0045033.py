def load(cls, pkid_or_path=None):
        """
        Load a container object from a persistent location or file path.

        Args:
            pkid_or_path: Integer pkid corresponding to the container table or file path

        Returns:
            container: The saved container object
        """
        path = pkid_or_path
        if isinstance(path, (int, np.int32, np.int64)):
            raise NotImplementedError('Lookup via CMS not implemented.')
        elif not os.path.isfile(path):
            raise FileNotFoundError('File {} not found.'.format(path))
        kwargs = {}
        fields = defaultdict(dict)
        with pd.HDFStore(path) as store:
            for key in store.keys():
                if 'kwargs' in key:
                    kwargs.update(store.get_storer(key).attrs.metadata)
                elif "FIELD" in key:
                    name, dname = "_".join(key.split("_")[1:]).split("/")
                    dname = dname.replace('values', '')
                    fields[name][dname] = store[key]
                else:
                    name = str(key[1:])
                    kwargs[name] = store[key]
        for name, field_data in fields.items():
            fps = field_data.pop('data')
            kwargs[name] = Field(fps, field_values=[field_data[str(arr)] for arr in
                                                    sorted(map(int, field_data.keys()))])
        return cls(**kwargs)
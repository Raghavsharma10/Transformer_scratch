def save(self, path):
        """
        Saves Datamat to path.

        Parameters:
            path : string
                Absolute path of the file to save to.
        """
        f = h5py.File(path, 'w')
        try:
            fm_group = f.create_group('Datamat')
            for field in self.fieldnames():
                try:
                    fm_group.create_dataset(field, data = self.__dict__[field])
                except (TypeError,) as e:
                    # Assuming field is an object array that contains dicts which
                    # contain numpy arrays as values
                    sub_group = fm_group.create_group(field)
                    for i, d in enumerate(self.__dict__[field]):
                        index_group = sub_group.create_group(str(i))
                        print((field, d))
                        for key, value in list(d.items()):
                            index_group.create_dataset(key, data=value)

            for param in self.parameters():
                fm_group.attrs[param]=self.__dict__[param]
        finally:
            f.close()
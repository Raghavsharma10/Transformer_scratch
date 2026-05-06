def get_file(cls, path):
        """Retrieves a file by storage name and fileid in the form of a path

        Path is expected to be ``storage_name/fileid``.
        """
        depot_name, file_id = path.split('/', 1)
        depot = cls.get(depot_name)
        return depot.get(file_id)
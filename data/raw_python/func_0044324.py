def scan_backends(self, backends):
        """
        From given backends create and return engine, filename and extension
        indexes.

        Arguments:
            backends (list): List of backend engines to scan. Order does matter
                since resulted indexes are stored in an ``OrderedDict``. So
                discovering will stop its job if it meets the first item.

        Returns:
            tuple: Engine, filename and extension indexes where:

            * Engines are indexed on their kind name with their backend object
              as value;
            * Filenames are indexed on their filename with engine kind name as
              value;
            * Extensions are indexed on their extension with engine kind name
              as value;
        """
        engines = OrderedDict()
        filenames = OrderedDict()
        extensions = OrderedDict()

        for item in backends:
            engines[item._kind_name] = item
            filenames[item._default_filename] = item._kind_name
            extensions[item._file_extension] = item._kind_name

        return engines, filenames, extensions
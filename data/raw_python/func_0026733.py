def init_from_file(cls,
                       catalog,
                       name=None,
                       path=None,
                       clean=False,
                       merge=True,
                       pop_schema=True,
                       ignore_keys=[],
                       compare_to_existing=True,
                       try_gzip=False,
                       filter_on={}):
        """Construct a new `Entry` instance from an input file.

        The input file can be given explicitly by `path`, or a path will
        be constructed appropriately if possible.

        Arguments
        ---------
        catalog : `astrocats.catalog.catalog.Catalog` instance
            The parent catalog object of which this entry belongs.
        name : str or 'None'
            The name of this entry, e.g. `SN1987A` for a `Supernova` entry.
            If no `path` is given, a path is constructed by trying to find
            a file in one of the 'output' repositories with this `name`.
            note: either `name` or `path` must be provided.
        path : str or 'None'
            The absolutely path of the input file.
            note: either `name` or `path` must be provided.
        clean : bool
            Whether special sanitization processing should be done on the input
            data.  This is mostly for input files from the 'internal'
            repositories.

        """
        if not catalog:
            from astrocats.catalog.catalog import Catalog
            log = logging.getLogger()
            catalog = Catalog(None, log)

        catalog.log.debug("init_from_file()")
        if name is None and path is None:
            err = ("Either entry `name` or `path` must be specified to load "
                   "entry.")
            log.error(err)
            raise ValueError(err)

        # If the path is given, use that to load from
        load_path = ''
        if path is not None:
            load_path = path
            name = ''
        # If the name is given, try to find a path for it
        else:
            repo_paths = catalog.PATHS.get_repo_output_folders()
            for rep in repo_paths:
                filename = cls.get_filename(name)
                newpath = os.path.join(rep, filename + '.json')
                if os.path.isfile(newpath):
                    load_path = newpath
                    break

        if load_path is None or not os.path.isfile(load_path):
            # FIX: is this warning worthy?
            return None

        # Create a new `Entry` instance
        new_entry = cls(catalog, name)

        # Check if .gz file
        if try_gzip and not load_path.endswith('.gz'):
            try_gzip = False

        # Fill it with data from json file
        new_entry._load_data_from_json(
            load_path,
            clean=clean,
            merge=merge,
            pop_schema=pop_schema,
            ignore_keys=ignore_keys,
            compare_to_existing=compare_to_existing,
            gzip=try_gzip,
            filter_on=filter_on)

        return new_entry
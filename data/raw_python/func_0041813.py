def register(cls, archive_table, engine):
        """
        :param archive_table: the model for the users archive table
        :param engine: the database engine
        :param version_col_names: strings which correspond to columns that versioning will pivot \
            around. These columns must have a unique constraint set on them.
        """
        version_col_names = cls.version_columns
        if not version_col_names:
            raise LogTableCreationError('Need to specify version cols in cls.version_columns')
        if cls.ignore_columns is None:
            cls.ignore_columns = set()
        cls.ignore_columns.add('version_id')
        version_cols = [getattr(cls, col_name, None) for col_name in version_col_names]

        cls._validate(engine, *version_cols)

        archive_table._validate(engine, *version_cols)
        cls.ArchiveTable = archive_table
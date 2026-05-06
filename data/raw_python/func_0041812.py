def _validate(cls, engine, *version_cols):
        """
        Validates the archive table.

        Validates the following criteria:
            - all version columns exist in the archive table
            - the python types of the user table and archive table columns are the same
            - a user_id column exists
            - there is a unique constraint on version and the other versioned columns from the
            user table

        :param engine: instance of :class:`~sqlalchemy.engine.Engine`
        :param *version_cols: instances of :class:`~InstrumentedAttribute` from
        the user table corresponding to the columns that versioning pivots around
        :raises: :class:`~LogTableCreationError`
        """
        cls._version_col_names = set()
        for version_column_ut in version_cols:
            # Make sure all version columns exist on this table
            version_col_name = version_column_ut.key
            version_column_at = getattr(cls, version_col_name, None)
            if not isinstance(version_column_at, InstrumentedAttribute):
                raise LogTableCreationError("Log table needs {} column".format(version_col_name))

            # Make sure the type of the user table and log table columns are the same
            version_col_at_t = version_column_at.property.columns[0].type.__class__
            version_col_ut_t = version_column_ut.property.columns[0].type.__class__
            if version_col_at_t != version_col_ut_t:
                raise LogTableCreationError(
                    "Type of column {} must match in log and user table".format(version_col_name)
                )
            cls._version_col_names.add(version_col_name)

        # Ensure user added a user_id column
        # TODO: should user_id column be optional?
        user_id = getattr(cls, 'user_id', None)
        if not isinstance(user_id, InstrumentedAttribute):
            raise LogTableCreationError("Log table needs user_id column")

        # Check the unique constraint on the versioned columns
        version_col_names = list(cls._version_col_names) + ['version_id']
        if not utils.has_constraint(cls, engine, *version_col_names):
            raise LogTableCreationError("There is no unique constraint on the version columns")
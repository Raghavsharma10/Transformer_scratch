def map_to(self, attrname, tablename=None, selectable=None, 
                    schema=None, base=None, mapper_args=util.immutabledict()):
        """Configure a mapping to the given attrname.

        This is the "master" method that can be used to create any 
        configuration.

        :param attrname: String attribute name which will be
          established as an attribute on this :class:.`.SQLSoup`
          instance.
        :param base: a Python class which will be used as the
          base for the mapped class. If ``None``, the "base"
          argument specified by this :class:`.SQLSoup`
          instance's constructor will be used, which defaults to
          ``object``.
        :param mapper_args: Dictionary of arguments which will
          be passed directly to :func:`.orm.mapper`.
        :param tablename: String name of a :class:`.Table` to be
          reflected. If a :class:`.Table` is already available,
          use the ``selectable`` argument. This argument is
          mutually exclusive versus the ``selectable`` argument.
        :param selectable: a :class:`.Table`, :class:`.Join`, or
          :class:`.Select` object which will be mapped. This
          argument is mutually exclusive versus the ``tablename``
          argument.
        :param schema: String schema name to use if the
          ``tablename`` argument is present.


        """
        if attrname in self._cache:
            raise SQLSoupError(
                "Attribute '%s' is already mapped to '%s'" % (
                attrname,
                class_mapper(self._cache[attrname]).mapped_table
            ))

        if tablename is not None:
            if not isinstance(tablename, basestring):
                raise ArgumentError("'tablename' argument must be a string."
                                    )
            if selectable is not None:
                raise ArgumentError("'tablename' and 'selectable' "
                                    "arguments are mutually exclusive")

            selectable = Table(tablename, 
                                        self._metadata, 
                                        autoload=True, 
                                        autoload_with=self.bind, 
                                        schema=schema or self.schema)
        elif schema:
            raise ArgumentError("'tablename' argument is required when "
                                "using 'schema'.")
        elif selectable is not None:
            if not isinstance(selectable, expression.FromClause):
                raise ArgumentError("'selectable' argument must be a "
                                    "table, select, join, or other "
                                    "selectable construct.")
        else:
            raise ArgumentError("'tablename' or 'selectable' argument is "
                                    "required.")

        if not selectable.primary_key.columns and not \
                             'primary_key' in mapper_args:
            if tablename:
                raise SQLSoupError(
                            "table '%s' does not have a primary "
                            "key defined" % tablename)
            else:
                raise SQLSoupError(
                            "selectable '%s' does not have a primary "
                            "key defined" % selectable)

        mapped_cls = _class_for_table(
            self.session,
            self.engine,
            selectable,
            base or self.base,
            mapper_args
        )
        self._cache[attrname] = mapped_cls
        return mapped_cls
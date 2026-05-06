def apply_driver_hacks(self, app, info, options):
        """Call before engine creation."""
        # Don't forget to apply hacks defined on parent object.
        super(SQLAlchemy, self).apply_driver_hacks(app, info, options)

        if info.drivername == 'sqlite':
            connect_args = options.setdefault('connect_args', {})

            if 'isolation_level' not in connect_args:
                # disable pysqlite's emitting of the BEGIN statement entirely.
                # also stops it from emitting COMMIT before any DDL.
                connect_args['isolation_level'] = None

            if not event.contains(Engine, 'connect', do_sqlite_connect):
                event.listen(Engine, 'connect', do_sqlite_connect)
            if not event.contains(Engine, 'begin', do_sqlite_begin):
                event.listen(Engine, 'begin', do_sqlite_begin)

            from sqlite3 import register_adapter

            def adapt_proxy(proxy):
                """Get current object and try to adapt it again."""
                return proxy._get_current_object()

            register_adapter(LocalProxy, adapt_proxy)

        elif info.drivername == 'postgresql+psycopg2':  # pragma: no cover
            from psycopg2.extensions import adapt, register_adapter

            def adapt_proxy(proxy):
                """Get current object and try to adapt it again."""
                return adapt(proxy._get_current_object())

            register_adapter(LocalProxy, adapt_proxy)

        elif info.drivername == 'mysql+pymysql':  # pragma: no cover
            from pymysql import converters

            def escape_local_proxy(val, mapping):
                """Get current object and try to adapt it again."""
                return converters.escape_item(
                    val._get_current_object(),
                    self.engine.dialect.encoding,
                    mapping=mapping,
                )

            converters.conversions[LocalProxy] = escape_local_proxy
            converters.encoders[LocalProxy] = escape_local_proxy
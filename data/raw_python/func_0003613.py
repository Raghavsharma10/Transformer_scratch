def make_middleware(cls, app, **options):
        """Creates the application WSGI middleware in charge of serving local files.

        A Depot middleware is required if your application wants to serve files from
        storages that don't directly provide and HTTP interface like
        :class:`depot.io.local.LocalFileStorage` and :class:`depot.io.gridfs.GridFSStorage`

        """
        from depot.middleware import DepotMiddleware
        mw = DepotMiddleware(app, **options)
        cls.set_middleware(mw)
        return mw
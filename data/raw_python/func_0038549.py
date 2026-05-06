def init_app(self, app):
        """Flask application initialization.

        :param app: The Flask application.
        :returns: The :class:`invenio_pages.ext.InvenioPages` instance
            initialized.
        """
        self.init_config(app)

        self.wrap_errorhandler(app)
        app.extensions['invenio-pages'] = _InvenioPagesState(app)

        return app.extensions['invenio-pages']
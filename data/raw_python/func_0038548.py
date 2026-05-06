def wrap_errorhandler(app):
        """Wrap error handler.

        :param app: The Flask application.
        """
        try:
            existing_handler = app.error_handler_spec[None][404][NotFound]
        except (KeyError, TypeError):
            existing_handler = None

        if existing_handler:
            app.error_handler_spec[None][404][NotFound] = \
                lambda error: handle_not_found(error, wrapped=existing_handler)
        else:
            app.error_handler_spec.setdefault(None, {}).setdefault(404, {})
            app.error_handler_spec[None][404][NotFound] = handle_not_found
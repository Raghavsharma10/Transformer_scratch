def _canvas_route(self, *args, **kwargs):
    """ Decorator for canvas route 
    """
    def outer(view_fn):
        @self.route(*args, **kwargs)
        def inner(*args, **kwargs):
            fn_args = getargspec(view_fn)
            try:
                idx = fn_args.args.index(_ARG_KEY)
            except ValueError:
                idx = -1

            if idx > -1:
                if 'error' in flask_request.args:
                    return redirect('%s?error=%s' % (
                        self.config.get('CANVAS_ERROR_URI', '/'),
                        flask_request.args.get('error')))

                if 'signed_request' not in flask_request.form:
                    self.logger.error('signed_request not in request.form')
                    abort(403)

                try:
                    _, decoded_data = _decode_signed_user(
                        *flask_request.form['signed_request'].split('.'))
                except ValueError as e:
                    self.logger.error(e.message)
                    abort(403)

                if 'oauth_token' not in decoded_data:
                    app.logger.info('unauthorized user, redirecting')
                    return _authorize()

                user = User(**decoded_data)

                if not app.config.get('CANVAS_SKIP_AUTH_CHECK', False) \
                    and not user.has_permissions():
                    self.logger.info(
                        'user does not have the required permission set.')
                    return _authorize()

                self.logger.info('all required permissions have been granted')
                args = args[:idx - 1] + (user,) + args[idx:]

            return view_fn(*args, **kwargs)
        return inner
    return outer
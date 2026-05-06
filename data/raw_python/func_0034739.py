def init_app(self, app, url='/hooks'):
        """Register the URL route to the application.

        :param app: the optional :class:`~flask.Flask` instance to
                register the extension
        :param url: the url that events will be posted to
        """
        app.config.setdefault('VALIDATE_IP', True)
        app.config.setdefault('VALIDATE_SIGNATURE', True)

        @app.route(url, methods=['POST'])
        def hook():
            if app.config['VALIDATE_IP']:
                if not is_github_ip(request.remote_addr):
                    raise Forbidden('Requests must originate from GitHub')

            if app.config['VALIDATE_SIGNATURE']:
                key = app.config.get('GITHUB_WEBHOOKS_KEY', app.secret_key)
                signature = request.headers.get('X-Hub-Signature')

                if hasattr(request, 'get_data'):
                    # Werkzeug >= 0.9
                    payload = request.get_data()
                else:
                    payload = request.data

                if not signature:
                    raise BadRequest('Missing signature')

                if not check_signature(signature, key, payload):
                    raise BadRequest('Wrong signature')

            event = request.headers.get('X-GitHub-Event')
            guid = request.headers.get('X-GitHub-Delivery')
            if not event:
                raise BadRequest('Missing header: X-GitHub-Event')
            elif not guid:
                raise BadRequest('Missing header: X-GitHub-Delivery')

            if hasattr(request, 'get_json'):
                # Flask >= 0.10
                data = request.get_json()
            else:
                data = request.json

            if event in self._hooks:
                return self._hooks[event](data, guid)
            else:
                return 'Hook not used\n'
def config_oauth(app):
    " Configure oauth support. "

    for name in PROVIDERS:
        config = app.config.get('OAUTH_%s' % name.upper())

        if not config:
            continue

        if not name in oauth.remote_apps:
            remote_app = oauth.remote_app(name, **config)

        else:
            remote_app = oauth.remote_apps[name]

        client_class = CLIENTS.get(name)
        client_class(app, remote_app)
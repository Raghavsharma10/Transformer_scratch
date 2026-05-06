def config_extensions(app):
    " Init application with extensions. "

    cache.init_app(app)
    db.init_app(app)
    main.init_app(app)
    collect.init_app(app)

    config_babel(app)
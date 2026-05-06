def config_babel(app):
    " Init application with babel. "

    babel.init_app(app)

    def get_locale():
        return request.accept_languages.best_match(app.config['BABEL_LANGUAGES'])
    babel.localeselector(get_locale)
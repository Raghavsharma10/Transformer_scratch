def run_gunicorn(application: WSGIHandler, gunicorn_module_name: str = 'gunicorn_prod'):
    """
    Runs gunicorn with a specified config.

    :param application: Django uwsgi application
    :param gunicorn_module_name: gunicorn settings module name
    :return: ``Application().run()``
    """
    from gunicorn.app.base import Application

    class DjangoApplication(Application):
        def init(self, parser, opts, args):
            cfg = self.get_config_from_module_name(gunicorn_module_name)
            clean_cfg = {}
            for k, v in cfg.items():
                # Ignore unknown names
                if k not in self.cfg.settings:
                    continue
                clean_cfg[k.lower()] = v
            return clean_cfg

        def load(self) -> WSGIHandler:
            return application

    return DjangoApplication().run()
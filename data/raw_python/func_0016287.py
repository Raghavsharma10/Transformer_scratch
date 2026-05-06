def ready(self):
        """Validate config and connect signals."""
        super(ElasticAppConfig, self).ready()
        _validate_config(settings.get_setting("strict_validation"))
        _connect_signals()
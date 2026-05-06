def get_backend_class():
    """Return reference to the configured backed class."""
    # this will (intentionally) blow up if the setting does not exist
    assert hasattr(settings, 'INBOUND_EMAIL_PARSER')
    assert getattr(settings, 'INBOUND_EMAIL_PARSER') is not None

    package, klass = settings.INBOUND_EMAIL_PARSER.rsplit('.', 1)
    module = import_module(package)
    return getattr(module, klass)
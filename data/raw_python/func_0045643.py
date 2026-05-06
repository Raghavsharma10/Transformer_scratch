def provider_factory(provider, **options):
    """ Factory function for DB Provider Concrete Classes
    """
    try:
        return {"tmdb": TMDb, "tvdb": TVDb}[provider.lower()](**options)
    except KeyError:
        msg = "Attempted to initialize non-existing DB Provider"
        log.error(msg)
        raise MapiException(msg)
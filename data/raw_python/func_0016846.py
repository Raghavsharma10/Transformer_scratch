def _create_defaults_source_provider(cube, data_source):
    """
    Create a DefaultsSourceProvider object. This provides default
    data sources for each array defined on the hypercube. The data sources
    may either by obtained from the arrays 'default' data source
    or the 'test' data source.
    """
    from montblanc.impl.rime.tensorflow.sources import (
        find_sources, DEFAULT_ARGSPEC)
    from montblanc.impl.rime.tensorflow.sources import constant_cache

    # Obtain default data sources for each array,
    # Just take from defaults if test data isn't specified
    staging_area_data_source = ('default' if not data_source == 'test'
                                                      else data_source)

    cache = True

    default_prov = DefaultsSourceProvider(cache=cache)

    # Create data sources on the source provider from
    # the cube array data sources
    for n, a in cube.arrays().iteritems():
        # Unnecessary for temporary arrays
        if 'temporary' in a.tags:
            continue

        # Obtain the data source
        data_source = a.get(staging_area_data_source)

        # Array marked as constant, decorate the data source
        # with a constant caching decorator
        if cache is True and 'constant' in a.tags:
            data_source = constant_cache(data_source)

        method = types.MethodType(data_source, default_prov)
        setattr(default_prov, n, method)

    def _sources(self):
        """
        Override the sources method to also handle lambdas that look like
        lambda s, c: ..., as defined in the config module
        """

        try:
            return self._sources
        except AttributeError:
            self._sources = find_sources(self, [DEFAULT_ARGSPEC] + [['s', 'c']])

        return self._sources

    # Monkey patch the sources method
    default_prov.sources = types.MethodType(_sources, default_prov)

    return default_prov
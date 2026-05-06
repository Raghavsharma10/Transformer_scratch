def get_es(**overrides):
    """Return a elasticsearch Elasticsearch object using settings
    from ``settings.py``.

    :arg overrides: Allows you to override defaults to create the
        ElasticSearch object. You can override any of the arguments
        isted in :py:func:`elasticutils.get_es`.

    For example, if you wanted to create an ElasticSearch with a
    longer timeout to a different cluster, you'd do:

    >>> from elasticutils.contrib.django import get_es
    >>> es = get_es(urls=['http://some_other_cluster:9200'], timeout=30)

    """
    defaults = {
        'urls': settings.ES_URLS,
        'timeout': getattr(settings, 'ES_TIMEOUT', 5)
        }

    defaults.update(overrides)
    return base_get_es(**defaults)
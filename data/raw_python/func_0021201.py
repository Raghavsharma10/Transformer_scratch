def es_version_check(f):
    """Decorator to check Elasticsearch version."""
    @wraps(f)
    def inner(*args, **kwargs):
        cluster_ver = current_search.cluster_version[0]
        client_ver = ES_VERSION[0]
        if cluster_ver != client_ver:
            raise click.ClickException(
                'Elasticsearch version mismatch. Invenio was installed with '
                'Elasticsearch v{client_ver}.x support, but the cluster runs '
                'Elasticsearch v{cluster_ver}.x.'.format(
                    client_ver=client_ver,
                    cluster_ver=cluster_ver,
                ))
        return f(*args, **kwargs)
    return inner
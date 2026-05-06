def get_public_cms_app_namespaces():
    """
    :return: a tuple() with all cms app namespaces
    """
    qs = Page.objects.public()
    qs = qs.exclude(application_namespace=None)
    qs = qs.order_by('application_namespace')

    try:
        application_namespaces = list(
            qs.distinct('application_namespace').values_list(
                'application_namespace', flat=True))
    except NotImplementedError:
        # If SQLite used:
        #   DISTINCT ON fields is not supported by this database backend
        application_namespaces = list(
            set(qs.values_list('application_namespace', flat=True)))

    application_namespaces.sort()

    return tuple(application_namespaces)
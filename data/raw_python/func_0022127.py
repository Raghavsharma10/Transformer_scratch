def opensearch_dispatch(request):
    """OpenSearch wrapper"""

    ctx = {
        'shortname': settings.REGISTRY_PYCSW['metadata:main']['identification_title'],
        'description': settings.REGISTRY_PYCSW['metadata:main']['identification_abstract'],
        'developer': settings.REGISTRY_PYCSW['metadata:main']['contact_name'],
        'contact': settings.REGISTRY_PYCSW['metadata:main']['contact_email'],
        'attribution': settings.REGISTRY_PYCSW['metadata:main']['provider_name'],
        'tags': settings.REGISTRY_PYCSW['metadata:main']['identification_keywords'].replace(',', ' '),
        'url': settings.SITE_URL.rstrip('/')
    }

    return render_to_response('search/opensearch_description.xml', ctx,
                              content_type='application/opensearchdescription+xml')
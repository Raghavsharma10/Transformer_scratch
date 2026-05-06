def csw_global_dispatch_by_catalog(request, catalog_slug):
    """pycsw wrapper for catalogs"""

    catalog = get_object_or_404(Catalog, slug=catalog_slug)

    if catalog:  # define catalog specific settings
        url = settings.SITE_URL.rstrip('/') + request.path.rstrip('/')
        return csw_global_dispatch(request, url=url, catalog_id=catalog.id)
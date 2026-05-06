async def viewers_js(request):
    '''
    Viewers determines the viewers installed based on settings, then uses the
    conversion infrastructure to convert all these JS files into a single JS
    bundle, that is then served. As with media, it will simply serve a cached
    version if necessary.
    '''
    # Generates single bundle as such:
    # BytesResource -> ViewerNodePackageBuilder -> nodepackage -> ... -> min.js
    response = singletons.server.response

    # Create a viewers resource, which is simply a JSON encoded description of
    # the viewers necessary for this viewers bundle.
    viewers_resource = singletons.viewers.get_resource()
    url_string = viewers_resource.url_string

    target_ts = TypeString('min.js')  # get a minified JS bundle
    target_resource = TypedResource(url_string, target_ts)

    if target_resource.cache_exists():
        return await response.file(target_resource.cache_path, headers={
            'Content-Type': 'application/javascript',
        })

    # Otherwise, does not exist, save this descriptor to cache and kick off
    # conversion process
    if not viewers_resource.cache_exists():
        viewers_resource.save()

    # Queue up a single function that will in turn queue up conversion process
    await singletons.workers.async_enqueue_sync(
        enqueue_conversion_path,
        url_string,
        str(target_ts),
        singletons.workers.enqueue_convert
    )

    return response.text(NOT_LOADED_JS, headers={
        'Content-Type': 'application/javascript',
    })
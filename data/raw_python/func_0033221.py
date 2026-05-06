async def convert_endpoint(url_string, ts, is_just_checking):
    '''
    Main logic for HTTP endpoint.
    '''
    response = singletons.server.response

    # Prep ForeignResource and ensure does not validate security settings
    singletons.settings
    foreign_res = ForeignResource(url_string)

    target_ts = TypeString(ts)
    target_resource = TypedResource(url_string, target_ts)

    # Send back cache if it exists
    if target_resource.cache_exists():
        if is_just_checking:
            return _just_checking_response(True, target_resource)
        return await response.file(target_resource.cache_path, headers={
            'Content-Type': target_ts.mimetype,
        })

    # Check if already downloaded. If not, queue up download.
    if not foreign_res.cache_exists():
        singletons.workers.enqueue_download(foreign_res)

    # Queue up a single function that will in turn queue up conversion
    # process
    singletons.workers.enqueue_sync(
        enqueue_conversion_path,
        url_string,
        str(target_ts),
        singletons.workers.enqueue_convert
    )

    if is_just_checking:
        return _just_checking_response(False, target_resource)

    # Respond with placeholder
    return singletons.placeholders.stream_response(target_ts, response)
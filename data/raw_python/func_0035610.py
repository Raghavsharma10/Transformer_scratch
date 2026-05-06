def cache_image_data(cache_dir, cache_key, uploader, *args, **kwargs):
    """ Call uploader and cache its results.
    """
    use_cache = True
    if "use_cache" in kwargs:
        use_cache = kwargs["use_cache"]
        del kwargs["use_cache"]

    json_path = None
    if cache_dir:
        json_path = os.path.join(cache_dir, "cached-img-%s.json" % cache_key)
        if use_cache and os.path.exists(json_path):
            LOG.info("Fetching %r from cache..." % (args,))
            try:
                with closing(open(json_path, "r")) as handle:
                    img_data = json.load(handle)

                return parts.Bunch([(key, parts.Bunch(val))
                    for key, val in img_data.items() # BOGUS pylint: disable=E1103
                ])
            except (EnvironmentError, TypeError, ValueError) as exc:
                LOG.warn("Problem reading cached data from '%s', ignoring cache... (%s)" % (json_path, exc))

    LOG.info("Copying %r..." % (args,))
    img_data = uploader(*args, **kwargs)

    if json_path:
        with closing(open(json_path, "w")) as handle:
            json.dump(img_data, handle)

    return img_data
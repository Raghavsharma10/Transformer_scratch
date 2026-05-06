def _split_js(media, domain):
    """
    Extract the local or external URLs from a Media object.
    """
    # Read internal property without creating new Media instance.
    if not media._js:
        return ImmutableMedia.empty_instance

    needs_local = domain == 'local'
    new_js = []
    for url in media._js:
        if needs_local == _is_local(url):
            new_js.append(url)

    if not new_js:
        return ImmutableMedia.empty_instance
    else:
        return Media(js=new_js)
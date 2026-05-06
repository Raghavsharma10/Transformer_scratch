def _split_css(media, domain):
    """
    Extract the local or external URLs from a Media object.
    """
    # Read internal property without creating new Media instance.
    if not media._css:
        return ImmutableMedia.empty_instance

    needs_local = domain == 'local'
    new_css = {}
    for medium, url in six.iteritems(media._css):
        if needs_local == _is_local(url):
            new_css.setdefault(medium, []).append(url)

    if not new_css:
        return ImmutableMedia.empty_instance
    else:
        return Media(css=new_css)
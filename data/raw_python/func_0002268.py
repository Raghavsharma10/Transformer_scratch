def register_frontend_media(request, media):
    """
    Add a :class:`~django.forms.Media` class to the current request.
    This will be rendered by the ``render_plugin_media`` template tag.
    """
    if not hasattr(request, '_fluent_contents_frontend_media'):
        request._fluent_contents_frontend_media = Media()

    add_media(request._fluent_contents_frontend_media, media)
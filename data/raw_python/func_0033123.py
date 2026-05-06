def get_frontpage_documents(context):
    """Returns the library favs that should be shown on the front page."""
    req = context.get('request')
    qs = Document.objects.published(req).filter(is_on_front_page=True)
    return qs
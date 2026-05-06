def if_url(context, url_name, yes, no):
    """
    Example:
        %li{ class:"{% if_url 'contacts.contact_read' 'active' '' %}" }
    """
    current = context["request"].resolver_match.url_name
    return yes if url_name == current else no
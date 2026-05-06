def get_comments_content_object(parser, token):
    """
    Get a limited set of comments for a given object.
    Defaults to a limit of 5. Setting the limit to -1 disables limiting.

    usage:

        {% get_comments_content_object for form_object as variable_name %}

    """
    keywords = token.contents.split()
    if len(keywords) != 5:
        raise template.TemplateSyntaxError(
            "'%s' tag takes exactly 2 arguments" % (keywords[0],))
    if keywords[1] != 'for':
        raise template.TemplateSyntaxError(
            "first argument to '%s' tag must be 'for'" % (keywords[0],))
    if keywords[3] != 'as':
        raise template.TemplateSyntaxError(
            "first argument to '%s' tag must be 'as'" % (keywords[0],))
    return GetCommentsContentObject(keywords[2], keywords[4])
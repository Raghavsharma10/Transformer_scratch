def get_newsentry_meta_description(newsentry):
    """Returns the meta description for the given entry."""
    if newsentry.meta_description:
        return newsentry.meta_description

    # If there is no seo addon found, take the info from the placeholders
    text = newsentry.get_description()

    if len(text) > 160:
        return u'{}...'.format(text[:160])
    return text
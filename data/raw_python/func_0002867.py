def ns_format(element, namespaced_string):
    """
    Provides a convenient method for adapting a tag or attribute name to
    use lxml's format. Use this for tags like ops:switch or attributes like
    xlink:href.
    """
    if ':' not in namespaced_string:
        print('This name contains no namespace, returning it unmodified: ' + namespaced_string)
        return namespaced_string
    namespace, name = namespaced_string.split(':')
    return '{' + element.nsmap[namespace] + '}' + name
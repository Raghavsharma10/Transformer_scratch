def compile_resource(resource):
    """
    Return compiled regex for resource matching
    """
    return re.compile("^" + trim_resource(re.sub(r":(\w+)", r"(?P<\1>[\w-]+?)",
                      resource)) + r"(\?(?P<querystring>.*))?$")
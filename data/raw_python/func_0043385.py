def get_namespace_and_tag(name):
        """
        Separates the namespace and tag from an element.

        :param str name: Tag.
        :returns: Namespace URI and Tag namespace.
        :rtype: tuple
        """

        if isinstance(name, str):
            if name[0] == "{":
                uri, ignore, tag = name[1:].partition("}")
            else:
                uri = None
                tag = name
        else:
            uri = None
            tag = None
        return uri, tag
def apply_args(self, **kwargs):
        # type: (**str) -> UrlPath
        """
        Apply formatting to each path node.

        This is used to apply a name to nodes (used to apply key names) eg:

        >>> a = UrlPath("foo", PathParam('{key_field}'), "bar")
        >>> b = a.apply_args(id="item_id")
        >>> b.format()
        'foo/{item_id}/bar'

        """
        def apply_format(node):
            if isinstance(node, PathParam):
                return PathParam(node.name.format(**kwargs), node.type, node.type_args)
            else:
                return node
        return UrlPath(*(apply_format(n) for n in self._nodes))
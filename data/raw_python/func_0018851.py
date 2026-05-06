def commentrepr(self) -> List[str]:
        """A list with comments for making string representations
        more informative.

        With option |Options.reprcomments| being disabled,
        |Variable.commentrepr| is empty.
        """
        if hydpy.pub.options.reprcomments:
            return [f'# {line}' for line in
                    textwrap.wrap(objecttools.description(self), 72)]
        return []
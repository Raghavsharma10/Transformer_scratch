def _pfp__add_child(self, name, child, stream=None):
        """Add a child to the Union field

        :name: The name of the child
        :child: A :class:`.Field` instance
        :returns: The resulting field
        """
        res = super(Union, self)._pfp__add_child(name, child)
        self._pfp__buff.seek(0, 0)
        child._pfp__build(stream=self._pfp__buff)
        size = len(self._pfp__buff.getvalue())
        self._pfp__buff.seek(0, 0)

        if stream is not None:
            curr_pos = stream.tell()
            stream.seek(curr_pos-size, 0)

        return res
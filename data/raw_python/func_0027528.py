def xml(self, indent=4, **kwargs):
        """
        :return: this node as XML text.

        Delegates to :meth:`write`
        """
        writer = StringIO()
        self.write(writer, indent=indent, **kwargs)
        return writer.getvalue()
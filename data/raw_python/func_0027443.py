def ignore_whitespace_text_nodes(cls, wrapped_node):
        """
        Find and delete any text nodes containing nothing but whitespace in
        in the given node and its descendents.

        This is useful for cleaning up excess low-value text nodes in a
        document DOM after parsing a pretty-printed XML document.
        """
        for child in wrapped_node.children:
            if child.is_text and child.value.strip() == '':
                child.delete()
            else:
                cls.ignore_whitespace_text_nodes(child)
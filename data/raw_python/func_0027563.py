def set_node_text(self, node, text):
        """
        Set text value as sole Text child node of element; any existing
        Text nodes are removed
        """
        # Remove any existing Text node children
        for child in self.get_node_children(node):
            if child.nodeType == xml.dom.Node.TEXT_NODE:
                self.remove_node_child(node, child, True)
        if text is not None:
            text_node = self.new_impl_text(text)
            self.add_node_child(node, text_node)
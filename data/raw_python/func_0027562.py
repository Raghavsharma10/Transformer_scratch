def get_node_text(self, node):
        """
        Return contatenated value of all text node children of this element
        """
        text_children = [n.nodeValue for n in self.get_node_children(node)
                         if n.nodeType == xml.dom.Node.TEXT_NODE]
        if text_children:
            return u''.join(text_children)
        else:
            return None
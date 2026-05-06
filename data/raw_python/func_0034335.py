def parent_of(self, name):
        """
        go to parent of node with name, and set as cur_node.  Useful
        for creating new paragraphs
       """
        if not self._in_tag(name):
            return

        node = self.cur_node
        while node.tag != name:
            node = node.getparent()
        self.cur_node = node.getparent()
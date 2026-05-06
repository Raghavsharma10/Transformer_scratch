def _is_last_child(self, tagname, attributes=None):
        """
        Check if last child of cur_node is tagname with attributes
        """
        children = self.cur_node.getchildren()
        if children:
            result = self._is_node(tagname, attributes, node=children[-1])
            return result

        return False
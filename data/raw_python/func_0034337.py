def _in_tag(self, tagname, attributes=None):
        """
        Determine if we are already in a certain tag.
        If we give attributes, make sure they match.
        """
        node = self.cur_node
        while not node is None:
            if node.tag == tagname:
                if attributes and node.attrib == attributes:
                    return True

                elif attributes:
                    return False

                return True

            node = node.getparent()
        return False
def remove_name(self):
        """
        Removes the name (short_description node) from the metadata node, if present.

        :return: True if the node is removed.  False is the node is node is not present.
        """
        short_description_node = self.metadata.find('short_description')
        if short_description_node is not None:
            self.metadata.remove(short_description_node)
            return True
        return False
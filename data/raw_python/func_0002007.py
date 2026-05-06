def remove_description(self):
        """
        Removes the description node from the metadata node, if present.

        :return: Returns True if the description node is removed. Returns False if the node is not present.
        """
        description_node = self.metadata.find('description')
        if description_node is not None:
            self.metadata.remove(description_node)
            return True
        return False
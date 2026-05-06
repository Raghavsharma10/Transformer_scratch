def update_description(self, description):
        """
        Update the description) of an IOC

        This creates the description node if it is not present.
        :param description: Value to set the description too
        :return: True
        """
        desc_node = self.metadata.find('description')
        if desc_node is None:
            log.debug('Could not find short description node for [{}].'.format(str(self.iocid)))
            log.debug('Creating & inserting the short description node')
            desc_node = ioc_et.make_description_node(description)
            insert_index = 0
            for child in self.metadata.getchildren():
                if child.tag == 'short_description':
                    index = self.metadata.index(child)
                    insert_index = index + 1
                    break
            self.metadata.insert(insert_index, desc_node)
        else:
            desc_node.text = description
        return True
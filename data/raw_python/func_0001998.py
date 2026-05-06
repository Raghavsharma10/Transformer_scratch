def update_name(self, name):
        """
        Update the name (short description) of an IOC

        This creates the short description node if it is not present.

        :param name: Value to set the short description too
        :return:
        """
        short_desc_node = self.metadata.find('short_description')
        if short_desc_node is None:
            log.debug('Could not find short description node for [{}].'.format(str(self.iocid)))
            log.debug('Creating & inserting the short description node')
            short_desc_node = ioc_et.make_short_description_node(name)
            self.metadata.insert(0, short_desc_node)
        else:
            short_desc_node.text = name
        return True
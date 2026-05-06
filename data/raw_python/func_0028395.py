def delNode(self, address):
        """
        Delete a node from the NodeServer

        :param node: Dictionary of node settings. Keys: address, name, node_def_id, primary, and drivers are required.
        """
        LOGGER.info('Removing node {}'.format(address))
        message = {
            'removenode': {
                'address': address
            }
        }
        self.send(message)
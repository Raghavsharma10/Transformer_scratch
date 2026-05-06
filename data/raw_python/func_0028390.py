def addNode(self, node):
        """
        Add a node to the NodeServer

        :param node: Dictionary of node settings. Keys: address, name, node_def_id, primary, and drivers are required.
        """
        LOGGER.info('Adding node {}({})'.format(node.name, node.address))
        message = {
            'addnode': {
                'nodes': [{
                    'address': node.address,
                    'name': node.name,
                    'node_def_id': node.id,
                    'primary': node.primary,
                    'drivers': node.drivers,
                    'hint': node.hint
                }]
            }
        }
        self.send(message)
def getNode(self, address):
        """
        Get Node by Address of existing nodes.
        """
        try:
            for node in self.config['nodes']:
                if node['address'] == address:
                    return node
            return False
        except KeyError:
            LOGGER.error('Usually means we have not received the config yet.', exc_info=True)
            return False
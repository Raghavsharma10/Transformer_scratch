def delNode(self, address):
        """
        Just send it along if requested, should be able to delete the node even if it isn't
        in our config anywhere. Usually used for normalization.
        """
        if address in self.nodes:
            del self.nodes[address]
        self.poly.delNode(address)
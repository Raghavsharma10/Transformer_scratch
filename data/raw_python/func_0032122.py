def createProduct(self, powerups):
        """
        Create a new L{Product} instance which confers the given
        powerups.

        @type powerups: C{list} of powerup item types

        @rtype: L{Product}
        @return: The new product instance.
        """
        types = [qual(powerup).decode('ascii')
                       for powerup in powerups]
        for p in self.store.parent.query(Product):
            for t in types:
                if t in p.types:
                    raise ValueError("%s is already included in a Product" % (t,))
        return Product(store=self.store.parent,
                       types=types)
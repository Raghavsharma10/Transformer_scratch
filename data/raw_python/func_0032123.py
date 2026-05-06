def coerceProduct(self, **kw):
        """
        Create a product and return a status string which should be part of a
        template.

        @param **kw: Fully qualified Python names for powerup types to
        associate with the created product.
        """
        self.original.createProduct(filter(None, kw.values()))
        return u'Created.'
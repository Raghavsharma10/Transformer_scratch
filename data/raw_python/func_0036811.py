def product(self, *products):
        r"""
            When search is called, it will limit the results to items in a Product.

            :param product: items passed in will be turned into a list
            :returns: :class:`Search`
        """
        for product in products:
            self._product.append(product)
        return self
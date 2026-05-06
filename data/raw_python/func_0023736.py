def order(self, field, order=None):
        """
        Set field and order set by arguments
        """
        if not order:
            order = ORDER.DESC
        self.url.order = (field, order)
        self.url.set_page(1)
        return self
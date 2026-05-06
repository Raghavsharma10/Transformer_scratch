def product(self):
        """Return product as human readable string."""
        if self.product_group == 14 and self.product_type == 3:
            return "KLF 200"
        return "Unknown Product: {}:{}".format(self.product_group, self.product_type)
def add_to_products(self, products=None, all_products=False):
        """
        Add user group to some product license configuration groups (PLCs), or all of them.
        :param products: list of product names the user should be added to
        :param all_products: a boolean meaning add to all (don't specify products in this case)
        :return: the Group, so you can do Group(...).add_to_products(...).add_users(...)
        """
        if all_products:
            if products:
                raise ArgumentError("When adding to all products, do not specify specific products")
            plist = "all"
        else:
            if not products:
                raise ArgumentError("You must specify products to which to add the user group")
            plist = {GroupTypes.productConfiguration.name: [product for product in products]}
        return self.append(add=plist)
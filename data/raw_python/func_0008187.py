def remove_from_products(self, products=None, all_products=False):
        """
        Remove user group from some product license configuration groups (PLCs), or all of them.
        :param products: list of product names the user group should be removed from
        :param all_products: a boolean meaning remove from all (don't specify products in this case)
        :return: the Group, so you can do Group(...).remove_from_products(...).add_users(...)
        """
        if all_products:
            if products:
                raise ArgumentError("When removing from all products, do not specify specific products")
            plist = "all"
        else:
            if not products:
                raise ArgumentError("You must specify products from which to remove the user group")
            plist = {GroupTypes.productConfiguration.name: [product for product in products]}
        return self.append(remove=plist)
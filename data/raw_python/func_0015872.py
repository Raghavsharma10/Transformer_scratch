def get_product_sets(self):
        """
        list all product sets for current user
        """
        # ensure we are using api url without a specific product set id
        api_url = super(ProductSetAPI, self).base_url
        return self.client.get(api_url)
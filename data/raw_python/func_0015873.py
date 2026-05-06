def delete_all_product_sets(self):
        """
        BE NOTICED: this will delete all product sets for current user
        """
        # ensure we are using api url without a specific product set id
        api_url = super(ProductSetAPI, self).base_url
        return self.client.delete(api_url)
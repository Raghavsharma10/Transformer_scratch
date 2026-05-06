def get_products(self, product_ids):
        """
        This function (and backend API) is being obsoleted. Don't use it anymore.
        """
        if self.product_set_id is None:
            raise ValueError('product_set_id must be specified')
        data = {'ids': product_ids}
        return self.client.get(self.base_url + '/products', json=data)
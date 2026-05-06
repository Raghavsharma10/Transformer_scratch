def get_params(self):
        """Get signature and params
        """
        params = {
            'key': self.get_app_key(),
            'uid': self.user_id,
            'widget': self.widget_code
        }

        products_number = len(self.products)

        if self.get_api_type() == self.API_GOODS:

            if isinstance(self.products, list):

                if products_number == 1:
                    product = self.products[0]

                    if isinstance(product, Product):
                        post_trial_product = None

                        if isinstance(product.get_trial_product(), Product):
                            post_trial_product = product
                            product = product.get_trial_product()

                        params['amount'] = product.get_amount()
                        params['currencyCode'] = product.get_currency_code()
                        params['ag_name'] = product.get_name()
                        params['ag_external_id'] = product.get_id()
                        params['ag_type'] = product.get_type()

                        if product.get_type() == Product.TYPE_SUBSCRIPTION:
                            params['ag_period_length'] = product.get_period_length()
                            params['ag_period_type'] = product.get_period_type()

                            if product.is_recurring():
                                params['ag_recurring'] = 1 if product.is_recurring() else 0

                                if post_trial_product:
                                    params['ag_trial'] = 1
                                    params['ag_post_trial_external_id'] = post_trial_product.get_id()
                                    params['ag_post_trial_period_length'] = post_trial_product.get_period_length()
                                    params['ag_post_trial_period_type'] = post_trial_product.get_period_type()
                                    params['ag_post_trial_name'] = post_trial_product.get_name()
                                    params['post_trial_amount'] = post_trial_product.get_amount()
                                    params['post_trial_currencyCode'] = post_trial_product.get_currency_code()
                    else:
                        self.append_to_errors('Not a Product instance')
                else:
                    self.append_to_errors('Only 1 product is allowed')

        elif self.get_api_type() == self.API_CART:
            index = 0

            for product in self.products:
                params['external_ids[' + str(index) + ']'] = product.get_id()
                if product.get_amount() > 0:
                    params['prices[' + str(index) + ']'] = product.get_amount()
                if product.get_currency_code() != '' and product.get_currency_code() is not None:
                    params['currencies[' + str(index) + ']'] = product.get_currency_code()
                index += 1

        params['sign_version'] = signature_version = str(self.get_default_widget_signature())

        if not self.is_empty(self.extra_params, 'sign_version'):
            signature_version = params['sign_version'] = str(self.extra_params['sign_version'])

        params = self.array_merge(params, self.extra_params)

        params['sign'] = self.calculate_signature(params, self.get_secret_key(), int(signature_version))
        return params
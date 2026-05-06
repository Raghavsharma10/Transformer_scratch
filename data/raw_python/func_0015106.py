def generate_cart_upload_redirect_url(self, **kwargs):
        """https://www.sandbox.paypal.com/webscr
            ?cmd=_cart
            &upload=1
        """
        required_vals = ('business', 'item_name_1', 'amount_1', 'quantity_1')
        self._check_required(required_vals, **kwargs)
        url = "%s?cmd=_cart&upload=1" % self.config.PAYPAL_URL_BASE
        additional = self._encode_utf8(**kwargs)
        additional = urlencode(additional)
        return url + "&" + additional
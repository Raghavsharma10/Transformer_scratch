def _parse_special_fields(self, data):
        """
        Helper method that parses special fields to Python objects

        :param data: response from Monzo API request
        :type data: dict
        """
        self.created = parse_date(data.pop('created'))

        if data.get('settled'):  # Not always returned
            self.settled = parse_date(data.pop('settled'))

        # Merchant field can contain either merchant ID or the whole object
        if (data.get('merchant') and
                not isinstance(data['merchant'], six.text_type)):
            self.merchant = MonzoMerchant(data=data.pop('merchant'))
def get_balance(self):
        """Check the balance fot this account.
           Returns a dictionary containing:
           account_type: The account type
           balance: The balance remaining on the account
           currency: The currency used for the account balance. Assume GBP in not set"""

        xml_root = self.__init_xml('Balance')

        response = clockwork_http.request(BALANCE_URL, etree.tostring(xml_root, encoding='utf-8'))
        data_etree = etree.fromstring(response['data'])

        err_desc = data_etree.find('ErrDesc')
        if err_desc is not None:
            raise clockwork_exceptions.ApiException(err_desc.text, data_etree.find('ErrNo').text)

        result = {}
        result['account_type'] = data_etree.find('AccountType').text
        result['balance'] = data_etree.find('Balance').text
        result['currency'] = data_etree.find('Currency').text
        return result
def get_cash_balance(self):
        """
        Returns the account cash balance available for investing

        Returns
        -------
        float
            The cash balance in your account.
        """
        cash = False
        try:
            response = self.session.get('/browse/cashBalanceAj.action')
            json_response = response.json()

            if self.session.json_success(json_response):
                self.__log('Cash available: {0}'.format(json_response['cashBalance']))
                cash_value = json_response['cashBalance']

                # Convert currency to float value
                # Match values like $1,000.12 or 1,0000$
                cash_match = re.search('^[^0-9]?([0-9\.,]+)[^0-9]?', cash_value)
                if cash_match:
                    cash_str = cash_match.group(1)
                    cash_str = cash_str.replace(',', '')
                    cash = float(cash_str)
            else:
                self.__log('Could not get cash balance: {0}'.format(response.text))

        except Exception as e:
            self.__log('Could not get the cash balance on the account: Error: {0}\nJSON: {1}'.format(str(e), response.text))
            raise e

        return cash
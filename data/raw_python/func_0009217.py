def __place_order(self, token):
        """
        Use the struts token to place the order.

        Parameters
        ----------
        token : string
            The struts token received from the place order page

        Returns
        -------
        int
            The completed order ID.
        """
        order_id = 0
        response = None

        if not token or token['value'] == '':
            raise LendingClubError('The token parameter is False, None or unknown.')

        # Process order confirmation page
        try:
            # Place the order
            payload = {}
            if token:
                payload['struts.token.name'] = token['name']
                payload[token['name']] = token['value']
            response = self.lc.session.post('/portfolio/orderConfirmed.action', data=payload)

            # Process HTML for the order ID
            html = response.text
            soup = BeautifulSoup(html, 'html5lib')

            # Order num
            order_field = soup.find(id='order_id')
            if order_field:
                order_id = int(order_field['value'])

            # Did not find an ID
            if order_id == 0:
                self.__log('An investment order was submitted, but a confirmation ID could not be determined')
                raise LendingClubError('No order ID was found when placing the order.', response)
            else:
                return order_id

        except Exception as e:
            raise LendingClubError('Could not place the order: {0}'.format(str(e)), response)
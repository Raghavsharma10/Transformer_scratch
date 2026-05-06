def place_order(self, quantity, symbol, price):
        """Place an order."""
        if price < 0:
            raise Exception("Price must be positive.")

        endpoint = "order"
        # Generate a unique clOrdID with our prefix so we can identify it.
        clOrdID = self.orderIDPrefix + base64.b64encode(uuid.uuid4().bytes).decode('ascii').rstrip('=\n')
        postdict = {
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'clOrdID': clOrdID
        }
        return self._curl_bitmex(api=endpoint, postdict=postdict, verb="POST")
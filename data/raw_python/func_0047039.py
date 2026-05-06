def open_orders(self, symbol=None):
        """Get open orders via HTTP. Used on close to ensure we catch them all."""
        api = "order"
        query = {'ordStatus.isTerminated': False }
        if symbol != None:
            query['symbol'] =symbol
        orders = self._curl_bitmex(
            api=api,
            query={'filter': json.dumps(query)},
            verb="GET"
        )
        return orders
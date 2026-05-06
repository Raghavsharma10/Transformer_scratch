def cancel(self, orderID):
        """Cancel an existing order."""
        api = "order"
        postdict = {
            'orderID': orderID,
        }
        return self._curl_bitmex(api=api, postdict=postdict, verb="DELETE")
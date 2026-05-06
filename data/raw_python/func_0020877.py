def get_purchase(self, purchase_id, purchase_key='sid'):
        """
        Retrieve information about a purchase using the system's unique ID or a client's ID
        @param id_: a string that represents a unique_id or an extid.
        @param key: a string that is either 'sid' or 'extid'.
        """
        data = {'purchase_id': purchase_id,
                'purchase_key': purchase_key}
        return self.api_get('purchase', data)
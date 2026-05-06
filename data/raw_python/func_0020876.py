def purchase(self, email, items=None, incomplete=None, message_id=None, options=None, extid=None):
        """
        Record that a user has made a purchase, or has added items to their purchase total.
        http://docs.sailthru.com/api/purchase
        @param email: Email string
        @param items: list of item dictionary with keys: id, title, price, qty, and url
        @param message_id: message_id string
        @param extid: external ID to track purchases
        @param options: other options that can be set as per the API documentation
        """
        items = items or {}
        options = options or {}
        data = options.copy()
        data['email'] = email
        data['items'] = items
        if incomplete is not None:
            data['incomplete'] = incomplete
        if message_id is not None:
            data['message_id'] = message_id
        if extid is not None:
            data['extid'] = extid
        return self.api_post('purchase', data)
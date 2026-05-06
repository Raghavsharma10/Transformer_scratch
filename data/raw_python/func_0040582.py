def on_data(self, raw_data):
        """Called when raw data is received from connection.

        Override this method if you wish to manually handle
        the stream data. Return False to stop stream and close connection.
        """
        data = json.loads(raw_data)

        message_type = data['meta'].get('type')
        prepare_method = 'prepare_%s' % (message_type)
        args = getattr(self, prepare_method, self.prepare_fallback)(data.get('data'))

        method_name = 'on_%s' % (message_type,)
        func = getattr(self, method_name, self.on_fallback)

        func(*args, meta=StreamingMeta.from_response_data(data.get('meta'), self.api))
def set_field_value(self, field_name, value):
        """ Set value of response field named `field_name`.

        If response contains single item, its field is set.
        If response contains multiple items, all the items in response
        are edited.
        To edit response meta(e.g. 'count') edit response directly at
        `event.response`.

        :param field_name: Name of response field value of which should
            be set.
        :param value: Value to be set.
        """
        if self.response is None:
            return

        if 'data' in self.response:
            items = self.response['data']
        else:
            items = [self.response]

        for item in items:
            item[field_name] = value
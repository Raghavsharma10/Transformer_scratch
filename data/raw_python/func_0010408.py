def parse_response(self, response):
        """
        Evaluates the action-call response from a FritzBox.
        The response is a xml byte-string.
        Returns a dictionary with the received arguments-value pairs.
        The values are converted according to the given data_types.
        TODO: boolean and signed integers data-types from tr64 responses
        """
        result = {}
        root = etree.fromstring(response)
        for argument in self.arguments.values():
            try:
                value = root.find('.//%s' % argument.name).text
            except AttributeError:
                # will happen by searching for in-parameters and by
                # parsing responses with status_code != 200
                continue
            if argument.data_type.startswith('ui'):
                try:
                    value = int(value)
                except ValueError:
                    # should not happen
                    value = None
            result[argument.name] = value
        return result
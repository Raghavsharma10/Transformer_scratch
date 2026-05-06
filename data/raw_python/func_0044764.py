def value_from_datadict(self, data, files, name):
        """Ensure the payload is a list of values. In the case of a sub
        form, we need to ensure the data is returned as a list and not a
        dictionary.

        When a dict is found in the given data, we need to ensure the data
        is converted to a list perseving the field order.

        """
        if name in data:
            payload = data.get(name)
            if isinstance(payload, (dict,)):
                # Make sure we get the data in the correct roder
                return [payload.get(f.name) for f in self.fields]
            return payload
        return super(FormFieldWidget, self).value_from_datadict(data, files, name)
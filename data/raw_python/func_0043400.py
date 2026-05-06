def clean_uri(self):
        """
        Validates the URI
        """
        if self.instance.fixed:
            return self.instance.uri

        uri = self.cleaned_data['uri']
        # todo: URI validation
        return uri
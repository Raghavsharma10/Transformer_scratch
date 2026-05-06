def content_type(self):
        """
        Returns the content type of the request in all cases where it is
        different than a submitted form - application/x-www-form-urlencoded
        """
        type_formencoded = "application/x-www-form-urlencoded"

        ctype = self.request.META.get('CONTENT_TYPE', type_formencoded)

        if type_formencoded in ctype:
            return None

        return ctype
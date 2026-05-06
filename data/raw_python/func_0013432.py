def obfuscate(cls, idStr):
        """
        Mildly obfuscates the specified ID string in an easily reversible
        fashion. This is not intended for security purposes, but rather to
        dissuade users from depending on our internal ID structures.
        """
        return unicode(base64.urlsafe_b64encode(
            idStr.encode('utf-8')).replace(b'=', b''))
def from_text_code(cls, email, result_text_code):
        """
        Alternative method to create an instance of VerifiedEmail object from a text code.
        :param str email: Email address.
        :param str result_text_code: A result of verification represented by text (e.g. valid, unknown).
        :return: An instance of object.
        """
        result_code = cls.result_text_codes[result_text_code]
        return cls(email, result_code)
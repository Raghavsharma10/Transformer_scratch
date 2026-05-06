def ipa_chars(self, value):
        """
        Set the list of IPAChar objects composing the IPA string

        :param list value: list of IPAChar objects
        """
        if value is None:
            self.__ipa_chars = []
        else:
            if is_list_of_ipachars(value):
                self.__ipa_chars = value
            else:
                raise TypeError("ipa_chars only accepts a list of IPAChar objects")
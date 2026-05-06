def tr(self, subdomain: str, string_to_translate: str = "") -> str:
        """Returns translation of string passed.

        :param str subdomain: subpart of strings dictionary.
         Must be one of self.translations.keys() i.e. 'restrictions'
        :param str string_to_translate: string you want to translate
        """
        if subdomain not in self.translations.keys():
            raise ValueError(
                "'{}' is not a correct subdomain."
                " Must be one of {}".format(subdomain, self.translations.keys())
            )
        else:
            pass
        # translate
        str_translated = self.translations.get(
            subdomain, {"error": "Subdomain not found: {}".format(subdomain)}
        ).get(string_to_translate, "String not found")

        # end of method
        return str_translated
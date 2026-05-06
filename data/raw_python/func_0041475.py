def get_template_language(self, file_):
        """
        Return the template language
        Every template file must end in
        with the language code, and the
        code must match the ISO_6301 lang code
        https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
        valid examples:

        account_created_pt.html
        payment_created_en.txt
        """
        stem = Path(file_).stem
        language_code = stem.split('_')[-1:][0]
        if len(language_code) != 2:
            # TODO naive and temp implementation
            # check if the two chars correspond to one of the
            # available languages
            raise Exception('Template file `%s` must end in ISO_639-1 language code.' % file_)
        return language_code.lower()
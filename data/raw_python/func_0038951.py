def fields_need_translation(self, elem, destination_lang):
        """
        Detect if the tuple needs translation and which fields has to be translated
        :param elem
        :param destination_lang:
        :return:
        """

        fields = self._get_translated_field_names(elem)
        elem_langs = elem.get_available_languages()
        # if we don't have a translation for the destination lang we have to include the tuple
        if destination_lang not in elem_langs:
            return fields

        # we have the translation, we decide which fields we need to translate. we have to get the translation first
        translation = get_translation(elem, destination_lang)
        result = []
        for field in fields:
            value = getattr(translation, field, '')
            if not value or value.strip() == '':
                result.append(field)

        return result
def add_item_languages(self, item, languages):
        """
        Update the TransItemLanguage model with the selected languages

        :param item:
        :param languages:
        :return:
        """
        # get the langs we have to add to the TransModelLanguage
        qs = TransLanguage.objects.filter(code__in=languages)
        new_langs = [lang for lang in qs]
        if not new_langs:
            return

        ct_item = ContentType.objects.get_for_model(item)
        item_lan, created = TransItemLanguage.objects.get_or_create(content_type_id=ct_item.id, object_id=item.id)
        item_lan.languages.add(*new_langs)
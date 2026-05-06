def remove_item_languages(self, item, languages):
        """
        delete the selected languages from the TransItemLanguage model

        :param item:
        :param languages:
        :return:
        """
        # get the langs we have to add to the TransModelLanguage
        qs = TransLanguage.objects.filter(code__in=languages)
        remove_langs = [lang for lang in qs]
        if not remove_langs:
            return

        ct_item = ContentType.objects.get_for_model(item)
        item_lan, created = TransItemLanguage.objects.get_or_create(content_type_id=ct_item.id, object_id=item.id)
        for lang in remove_langs:
            item_lan.languages.remove(lang)
        if item_lan.languages.count() == 0:
            item_lan.delete()
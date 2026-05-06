def get_languages_from_item(ct_item, item):
        """
        Get the languages configured for the current item
        :param ct_item:
        :param item:
        :return:
        """
        try:
            item_lan = TransItemLanguage.objects.filter(content_type__pk=ct_item.id, object_id=item.id).get()
            languages = [lang.code for lang in item_lan.languages.all()]
            return languages
        except TransItemLanguage.DoesNotExist:
            return []
def create_translations_for_item_and_its_children(self, item, languages=None):
        """
        Creates the translations from an item and defined languages and return the id's of the created tasks

        :param item: (master)
        :param languages:
        :return:
        """

        if not self.master:
            self.set_master(item)

        if not languages:
            languages = self.get_languages()

        result_ids = []

        # first process main object
        fields = self._get_translated_field_names(item)
        tasks = self.create_from_item(languages, item, fields)
        if tasks:
            result_ids += [task.pk for task in tasks]

        # then process child objects from main
        children = self.get_translatable_children(item)
        for child in children:
            fields = self._get_translated_field_names(child)
            tasks = self.create_from_item(languages, child, fields)
            if tasks:
                result_ids += [task.pk for task in tasks]

        return result_ids
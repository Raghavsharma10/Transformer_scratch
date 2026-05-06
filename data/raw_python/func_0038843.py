def get_languages(self, include_main=False):
        """
        Get all the languages except the main.

        Try to get in order:
            1.- item languages
            2.- model languages
            3.- application model languages
            # 4.- default languages

        :param master:
        :param include_main:
        :return:
        """

        if not self.master:
            raise Exception('TransManager - No master set')

        item_languages = self.get_languages_from_item(self.ct_master, self.master)

        languages = self.get_languages_from_model(self.ct_master.app_label, self.ct_master.model)
        if not languages:
            languages = self.get_languages_from_application(self.ct_master.app_label)
            # if not languages:
            #     languages = self.get_languages_default()

        if not include_main:
            main_language = self.get_main_language()
            if main_language in languages:
                languages.remove(main_language)

        return list(set(item_languages + languages))
def get_languages_from_model(app_label, model_label):
        """
        Get the languages configured for the current model

        :param model_label:
        :param app_label:
        :return:
        """
        try:
            mod_lan = TransModelLanguage.objects.filter(model='{} - {}'.format(app_label, model_label)).get()
            languages = [lang.code for lang in mod_lan.languages.all()]
            return languages
        except TransModelLanguage.DoesNotExist:
            return []
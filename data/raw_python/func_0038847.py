def get_languages_from_application(app_label):
        """
        Get the languages configured for the current application

        :param app_label:
        :return:
        """
        try:
            mod_lan = TransApplicationLanguage.objects.filter(application=app_label).get()
            languages = [lang.code for lang in mod_lan.languages.all()]
            return languages
        except TransApplicationLanguage.DoesNotExist:
            return []
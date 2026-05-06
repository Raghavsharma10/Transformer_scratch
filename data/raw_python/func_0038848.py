def get_main_language():
        """
        returns the main language
        :return:
        """
        try:
            main_language = TransLanguage.objects.filter(main_language=True).get()
            return main_language.code
        except TransLanguage.DoesNotExist:
            return TM_DEFAULT_LANGUAGE_CODE
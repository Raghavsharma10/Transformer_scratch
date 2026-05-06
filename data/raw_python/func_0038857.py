def get_translation_from_instance(instance, lang):
        """
        Get the translation from the instance in a specific language, hits the db

        :param instance:
        :param lang:
        :return:
        """
        try:
            translation = get_translation(instance, lang)
        except (AttributeError, ObjectDoesNotExist):
            translation = None
        return translation
def template_exists_db(self, template):
        """
        Receives a template and checks if it exists in the database
        using the template name and language
        """
        name = utils.camel_to_snake(template[0]).upper()
        language = utils.camel_to_snake(template[3])
        try:
            models.EmailTemplate.objects.get(name=name, language=language)
        except models.EmailTemplate.DoesNotExist:
            return False
        return True
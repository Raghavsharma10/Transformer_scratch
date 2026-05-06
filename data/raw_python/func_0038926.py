def save(self, force_insert=False, force_update=False, using=None, update_fields=None):
        """
        Overwrite of the save method in order that when setting the language
        as main we deactivate any other model selected as main before

        :param force_insert:
        :param force_update:
        :param using:
        :param update_fields:
        :return:
        """
        super().save(force_insert, force_update, using, update_fields)
        if self.main_language:
            TransLanguage.objects.exclude(pk=self.pk).update(main_language=False)
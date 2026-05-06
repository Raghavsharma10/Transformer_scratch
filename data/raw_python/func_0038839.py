def update_task(self, differences):
        """
        Updates a task as done if we have a new value for this alternative language

        :param differences:
        :return:
        """

        self.log('differences UPDATING: {}'.format(differences))

        object_name = '{} - {}'.format(self.app_label, self.instance.master._meta.verbose_name)
        lang = self.instance.language_code
        object_pk = self.instance.master.pk

        for field in differences:
            value = getattr(self.instance, field)
            if value is None or value == '':
                continue
            try:
                TransTask.objects.filter(
                    language__code=lang, object_field=field, object_name=object_name, object_pk=object_pk
                ).update(done=True, date_modification=datetime.now(), object_field_value_translation=value)
                self.log('MARKED TASK AS DONE')
            except TransTask.DoesNotExist:
                self.log('error MARKING TASK AS DONE: {} - {} - {} - {}'.format(lang, field, object_name, object_pk))
def get_previous_and_current_values(self, instance):
        """
        Obtain the previous and actual values and compares them
        in order to detect which fields has changed

        :param instance:
        :param translation:
        :return:
        """
        translated_field_names = self._get_translated_field_names(instance.master)
        if instance.pk:
            try:
                previous_obj = instance._meta.model.objects.get(pk=instance.pk)
                previous_values = self.get_obj_values(previous_obj, translated_field_names)
            except ObjectDoesNotExist:
                previous_values = {}
        else:
            previous_values = {}
        current_values = self.get_obj_values(instance, translated_field_names)
        return previous_values, current_values
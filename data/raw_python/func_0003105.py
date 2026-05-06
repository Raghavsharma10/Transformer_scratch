def get_status_key(self, instance):
        """Generates a key used to set a status on a field"""
        key_id = "inst_%s" % id(instance) if instance.pk is None else instance.pk
        return "%s.%s-%s-%s" % (instance._meta.app_label,
                                get_model_name(instance),
                                key_id,
                                self.field.name)
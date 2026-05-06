def label_for_value(self, value, key=None):
        """
        Looks up the current value of the field and returns
        a unicode representation. Default implementation does a lookup
        on the target model and if a match is found calls force_unicode
        on that object. Otherwise a blank string is returned.
        """
        if not key:
            key = self.rel.get_related_field().name

        if value is not None:
            try:
                obj = self.model._default_manager.using(self.db).get(**{key: value})
                return force_unicode(obj)
            except (ValueError, self.model.DoesNotExist):
                return ''
        return ''
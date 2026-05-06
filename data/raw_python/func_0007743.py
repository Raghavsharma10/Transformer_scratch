def _register_elements(self, elements):
        """ Takes elements from the metadata class and creates a base model for all backend models .
        """
        self.elements = elements

        for key, obj in elements.items():
            obj.contribute_to_class(self.metadata, key)

        # Create the common Django fields
        fields = {}
        for key, obj in elements.items():
            if obj.editable:
                field = obj.get_field()
                if not field.help_text:
                    if key in self.bulk_help_text:
                        field.help_text = self.bulk_help_text[key]
                fields[key] = field

        # 0. Abstract base model with common fields
        base_meta = type('Meta', (), self.original_meta)
        class BaseMeta(base_meta):
            abstract = True
            app_label = 'seo'
        fields['Meta'] = BaseMeta
        # Do we need this?
        fields['__module__'] = __name__ #attrs['__module__']
        self.MetadataBaseModel = type('%sBase' % self.name, (models.Model,), fields)
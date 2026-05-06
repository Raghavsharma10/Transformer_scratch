def get_entity_kind(self, model_obj):
        """
        Returns a tuple for a kind name and kind display name of an entity.
        By default, uses the app_label and model of the model object's content
        type as the kind.
        """
        model_obj_ctype = ContentType.objects.get_for_model(self.queryset.model)
        return (u'{0}.{1}'.format(model_obj_ctype.app_label, model_obj_ctype.model), u'{0}'.format(model_obj_ctype))
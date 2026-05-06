def get_for_obj(self, entity_model_obj):
        """
        Given a saved entity model object, return the associated entity.
        """
        return self.get(entity_type=ContentType.objects.get_for_model(
            entity_model_obj, for_concrete_model=False), entity_id=entity_model_obj.id)
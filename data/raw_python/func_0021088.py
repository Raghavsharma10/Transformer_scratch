def delete_for_obj(self, entity_model_obj):
        """
        Delete the entities associated with a model object.
        """
        return self.filter(
            entity_type=ContentType.objects.get_for_model(
                entity_model_obj, for_concrete_model=False), entity_id=entity_model_obj.id).delete(
            force=True)
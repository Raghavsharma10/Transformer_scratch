def delete_manager(self, instance):
        """
        Returns a queryset based on the related model's base manager (rather
        than the default manager, as returned by __get__). Used by
        Model.delete().
        """
        return self.create_manager(instance,
                                   self.rating_model._base_manager.__class__)
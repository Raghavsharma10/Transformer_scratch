def _create_model_by_type(self, type):
        """
        Create a new model instance by type.

        :rtype: Model
        """
        klass = None
        for cls in eloquent.orm.model.Model.__subclasses__():
            morph_class = cls.__morph_class__ or cls.__name__
            if morph_class == type:
                klass = cls
                break

        return klass()
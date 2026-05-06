def fname(self, model_or_obj_or_qs):
        """ 
        Return the field name on the :class:`Follow` model for ``model_or_obj_or_qs``.
        """
        if isinstance(model_or_obj_or_qs, QuerySet):
            _, fname = model_map[model_or_obj_or_qs.model]
        else:
            cls = model_or_obj_or_qs if inspect.isclass(model_or_obj_or_qs) else model_or_obj_or_qs.__class__
            _, fname = model_map[cls]
        return fname
def get_follows(self, model_or_obj_or_qs):
        """
        Returns all the followers of a model, an object or a queryset.
        """
        fname = self.fname(model_or_obj_or_qs)
        
        if isinstance(model_or_obj_or_qs, QuerySet):
            return self.filter(**{'%s__in' % fname: model_or_obj_or_qs})
        
        if inspect.isclass(model_or_obj_or_qs):
            return self.exclude(**{fname:None})

        return self.filter(**{fname:model_or_obj_or_qs})
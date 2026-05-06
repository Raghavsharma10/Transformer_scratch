def question_default_add_related_pks(self, obj):
        """Add related primary keys to a Question instance."""
        if not hasattr(obj, '_choice_pks'):
            obj._choice_pks = list(obj.choices.values_list('pk', flat=True))
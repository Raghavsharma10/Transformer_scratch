def choice_default_invalidator(self, obj):
        """Invalidated cached items when the Choice changes."""
        invalid = [('Question', obj.question_id, True)]
        for pk in obj.voters.values_list('pk', flat=True):
            invalid.append(('User', pk, False))
        return invalid
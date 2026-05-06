def choice_default_serializer(self, obj):
        """Convert a Choice to a cached instance representation."""
        if not obj:
            return None
        self.choice_default_add_related_pks(obj)
        return dict((
            ('id', obj.id),
            ('choice_text', obj.choice_text),
            self.field_to_json(
                'PK', 'question', model=Question, pk=obj.question_id),
            self.field_to_json(
                'PKList', 'voters', model=User, pks=obj._voter_pks)
        ))
def question_default_serializer(self, obj):
        """Convert a Question to a cached instance representation."""
        if not obj:
            return None
        self.question_default_add_related_pks(obj)
        return dict((
            ('id', obj.id),
            ('question_text', obj.question_text),
            self.field_to_json('DateTime', 'pub_date', obj.pub_date),
            self.field_to_json(
                'PKList', 'choices', model=Choice, pks=obj._choice_pks),
        ))
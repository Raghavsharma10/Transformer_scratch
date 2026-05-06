def user_default_serializer(self, obj):
        """Convert a User to a cached instance representation."""
        if not obj:
            return None
        self.user_default_add_related_pks(obj)
        return dict((
            ('id', obj.id),
            ('username', obj.username),
            self.field_to_json('DateTime', 'date_joined', obj.date_joined),
            self.field_to_json(
                'PKList', 'votes', model=Choice, pks=obj._votes_pks),
        ))
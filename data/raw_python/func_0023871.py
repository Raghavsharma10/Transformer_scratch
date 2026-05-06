def build_object(self, obj):
        """Override django-bakery to skip pages marked exclude_from_static"""
        if not obj.exclude_from_static:
            super(ShowPage, self).build_object(obj)
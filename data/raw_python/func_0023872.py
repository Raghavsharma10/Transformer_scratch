def build_object(self, obj):
        """Override django-bakery to skip talks that raise 403"""
        try:
            super(TalkView, self).build_object(obj)
        except PermissionDenied:
            # We cleanup the directory created
            self.unbuild_object(obj)
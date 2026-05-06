def get_object(self, *args, **kwargs):
        '''Only talk owners can see talks, unless they've been accepted'''
        object_ = super(TalkView, self).get_object(*args, **kwargs)
        if not object_.can_view(self.request.user):
            raise PermissionDenied
        return object_
def get_or_create(self, user, obj, **kwargs):
        """ 
        Almost the same as `FollowManager.objects.create` - behaves the same 
        as the normal `get_or_create` methods in django though. 

        Returns a tuple with the `Follow` and either `True` or `False`

        """
        if not self.is_following(user, obj):
            return self.create(user, obj, **kwargs), True
        return self.get_follows(obj).get(user=user), False
def is_following(self, user, obj):
        """ Returns `True` or `False` """
        if isinstance(user, AnonymousUser):
            return False        
        return 0 < self.get_follows(obj).filter(user=user).count()
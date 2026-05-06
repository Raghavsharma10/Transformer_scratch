def create(self, user, obj, **kwargs):
        """
        Create a new follow link between a user and an object
        of a registered model type.
        
        """
        follow = Follow(user=user)
        follow.target = obj
        follow.save()
        return follow
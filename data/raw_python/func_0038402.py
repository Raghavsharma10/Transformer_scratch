def has_object_permission(self, request, view, obj):
        """determines if requesting user has permissions for the object

        :param request: WSGI request object - where we get the user from
        :param view: the view calling for permission
        :param obj: the object in question
        :return: `bool`
        """
        # Give permission if we're not protecting this method
        if self.protected_methods and request.method not in self.protected_methods:
            return True

        user = getattr(request, "user", None)

        if not user or user.is_anonymous():
            return False

        if self.require_staff and not user.is_staff:
            return False

        # if they have higher-level privileges we can return true right now
        if user.has_perms(self.permissions):
            return True

        # no? ok maybe they're the author and have appropriate author permissions.
        authors_field = getattr(obj, self.authors_field, None)

        if not authors_field:
            return False

        if self.author_permissions and not user.has_perms(self.author_permissions):
            return False

        return user in authors_field.all()
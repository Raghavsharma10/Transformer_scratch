def has_permission(self, request, view):
        """If method is GET, user can access, if method is PUT or POST user must be a superuser.
        """
        has_permission = False
        if request.method == "GET" \
                or request.method in ["PUT", "POST", "DELETE"] \
                and request.user and request.user.is_superuser:
            has_permission = True
        return has_permission
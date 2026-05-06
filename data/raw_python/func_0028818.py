def get_comment_object(self):
        """
        NB: Overridden to remove dupe comment check for admins (necessary for
        canned responses)

        Return a new (unsaved) comment object based on the information in this
        form. Assumes that the form is already validated and will throw a
        ValueError if not.

        Does not set any of the fields that would come from a Request object
        (i.e. ``user`` or ``ip_address``).
        """
        if not self.is_valid():
            raise ValueError(
                "get_comment_object may only be called on valid forms")

        CommentModel = self.get_comment_model()
        new = CommentModel(**self.get_comment_create_data())

        user_model = get_user_model()
        try:
            user = user_model.objects.get(username=new.user_name)
            if not user.is_staff:
                new = self.check_for_duplicate_comment(new)
        except user_model.DoesNotExist:
            # post_molo_comment may have set the username to 'Anonymous'
            new = self.check_for_duplicate_comment(new)

        return new
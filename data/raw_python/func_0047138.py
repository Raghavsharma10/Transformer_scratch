def save_comment(self, comment_form, *args, **kwargs):
        """Pass through to provider CommentAdminSession.update_comment"""
        # Implemented from kitosid template for -
        # osid.resource.ResourceAdminSession.update_resource
        if comment_form.is_for_update():
            return self.update_comment(comment_form, *args, **kwargs)
        else:
            return self.create_comment(comment_form, *args, **kwargs)
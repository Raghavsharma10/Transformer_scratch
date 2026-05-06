def get_comments_by_search(self, comment_query, comment_search):
        """Pass through to provider CommentSearchSession.get_comments_by_search"""
        # Implemented from azosid template for -
        # osid.resource.ResourceSearchSession.get_resources_by_search_template
        if not self._can('search'):
            raise PermissionDenied()
        return self._provider_session.get_comments_by_search(comment_query, comment_search)
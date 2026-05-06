def get_books_by_comment(self, *args, **kwargs):
        """Pass through to provider CommentBookSession.get_books_by_comment"""
        # Implemented from kitosid template for -
        # osid.resource.ResourceBinSession.get_bins_by_resource
        catalogs = self._get_provider_session('comment_book_session').get_books_by_comment(*args, **kwargs)
        cat_list = []
        for cat in catalogs:
            cat_list.append(Book(self._provider_manager, cat, self._runtime, self._proxy))
        return BookList(cat_list)
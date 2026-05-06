def get_books(self):
        """Pass through to provider BookLookupSession.get_books"""
        # Implemented from kitosid template for -
        # osid.resource.BinLookupSession.get_bins_template
        catalogs = self._get_provider_session('book_lookup_session').get_books()
        cat_list = []
        for cat in catalogs:
            cat_list.append(Book(self._provider_manager, cat, self._runtime, self._proxy))
        return BookList(cat_list)
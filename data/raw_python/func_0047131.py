def save_book(self, book_form, *args, **kwargs):
        """Pass through to provider BookAdminSession.update_book"""
        # Implemented from kitosid template for -
        # osid.resource.BinAdminSession.update_bin
        if book_form.is_for_update():
            return self.update_book(book_form, *args, **kwargs)
        else:
            return self.create_book(book_form, *args, **kwargs)
def get_book_form(self, *args, **kwargs):
        """Pass through to provider BookAdminSession.get_book_form_for_update"""
        # Implemented from kitosid template for -
        # osid.resource.BinAdminSession.get_bin_form_for_update_template
        # This method might be a bit sketchy. Time will tell.
        if isinstance(args[-1], list) or 'book_record_types' in kwargs:
            return self.get_book_form_for_create(*args, **kwargs)
        else:
            return self.get_book_form_for_update(*args, **kwargs)
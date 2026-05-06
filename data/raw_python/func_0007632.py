def get_screen_pointers(self):
        """
        Returns the xcb_screen_t for every screen
        useful for other bindings
        """
        root_iter = lib.xcb_setup_roots_iterator(self._setup)

        screens = [root_iter.data]
        for i in range(self._setup.roots_len - 1):
            lib.xcb_screen_next(ffi.addressof((root_iter)))
            screens.append(root_iter.data)
        return screens
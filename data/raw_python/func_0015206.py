def create_scrolled_window(self, layout_manager, horizontal=Gtk.PolicyType.NEVER, vertical=Gtk.PolicyType.ALWAYS):
        """
        Function creates a scrolled window with layout manager
        """
        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.add(layout_manager)
        scrolled_window.set_policy(horizontal, vertical)
        return scrolled_window
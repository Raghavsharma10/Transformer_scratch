def create_label(self, name, justify=Gtk.Justification.CENTER, wrap_mode=True, tooltip=None):
        """
        The function is used for creating lable with HTML text
        """
        label = Gtk.Label()
        name = name.replace('|', '\n')
        label.set_markup(name)
        label.set_justify(justify)
        label.set_line_wrap(wrap_mode)
        if tooltip is not None:
            label.set_has_tooltip(True)
            label.connect("query-tooltip", self.parent.tooltip_queries, tooltip)
        return label
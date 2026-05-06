def stylize(self):
        """Apply theme style attributes to this instance and its children.

        This also causes a relayout to occur so that any changes in padding
        or other stylistic attributes may be handled.
        """
        # do children first in case parent needs to override their style
        for child in self.children:
            child.stylize()
        style = theme.current.get_dict(self)
        preserve_child = False
        try:
            preserve_child = getattr(theme.current, 'preserve_child')
        except:
            preserve_child = False

        for key, val in style.iteritems():
            kvc.set_value_for_keypath(self, key, val, preserve_child)
        self.layout()
def sprite_changes(cls, sprite):
        """Return a mapping of attributes to their initilization state."""
        retval = dict((x, cls.attribute_state(sprite.scripts, x)) for x in
                      (x for x in cls.ATTRIBUTES if x != 'background'))
        return retval
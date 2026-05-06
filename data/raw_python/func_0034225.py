def draw_key(self, surface, key):
        """Default drawing method for key. 

        Draw the key accordingly to it type.

        :param surface: Surface background should be drawn in.
        :param key: Target key to be drawn.
        """
        if isinstance(key, VSpaceKey):
            self.draw_space_key(surface, key)
        elif isinstance(key, VBackKey):
            self.draw_back_key(surface, key)
        elif isinstance(key, VUppercaseKey):
            self.draw_uppercase_key(surface, key)
        elif isinstance(key, VSpecialCharKey):
            self.draw_special_char_key(surface, key)
        else:
            self.draw_character_key(surface, key)
def draw_special_char_key(self, surface, key):
        """Default drawing method for special char key. Drawn as character key.

        :param surface: Surface background should be drawn in.
        :param key: Target key to be drawn.
        """
        key.value = u'#' 
        if key.is_activated():
            key.value = u'Ab'
        self.draw_character_key(surface, key, True)
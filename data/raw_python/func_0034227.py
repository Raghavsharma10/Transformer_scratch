def draw_uppercase_key(self, surface, key):
        """Default drawing method for uppercase key. Drawn as character key.

        :param surface: Surface background should be drawn in.
        :param key: Target key to be drawn.
        """
        key.value = u'\u21e7' 
        if key.is_activated():
            key.value = u'\u21ea'
        self.draw_character_key(surface, key, True)
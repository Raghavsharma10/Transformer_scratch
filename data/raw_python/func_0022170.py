def _get_pastel_colour(self, lighten=127):
        """
            Create a pastel colour hex colour string
        """
        def r():
            return random.randint(0, 128) + lighten
        return r(), r(), r()
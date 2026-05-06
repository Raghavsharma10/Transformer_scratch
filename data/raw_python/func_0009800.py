def level(self):
        """Get level from vera."""
        # Used for dimmers, curtains
        # Have seen formats of 10, 0.0 and "0%"!
        level = self.get_value('level')
        try:
            return int(float(level))
        except (TypeError, ValueError):
            pass
        try:
            return int(level.strip('%'))
        except (TypeError, AttributeError, ValueError):
            pass
        return 0
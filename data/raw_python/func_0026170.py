def _fetch(self, key):
        """Helper function to fetch values from owning section.

        Returns a 2-tuple: the value, and the section where it was found.
        """
        # switch off interpolation before we try and fetch anything !
        save_interp = self.section.main.interpolation
        self.section.main.interpolation = False

        # Start at section that "owns" this InterpolationEngine
        current_section = self.section
        while True:
            # try the current section first
            val = current_section.get(key)
            if val is not None and not isinstance(val, Section):
                break
            # try "DEFAULT" next
            val = current_section.get('DEFAULT', {}).get(key)
            if val is not None and not isinstance(val, Section):
                break
            # move up to parent and try again
            # top-level's parent is itself
            if current_section.parent is current_section:
                # reached top level, time to give up
                break
            current_section = current_section.parent

        # restore interpolation to previous value before returning
        self.section.main.interpolation = save_interp
        if val is None:
            raise MissingInterpolationOption(key)
        return val, current_section
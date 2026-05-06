def _clone(self, **attrs):
        """
        Makes a copy of an model instance.

        for every key in **attrs value will
        be set on the new instance.
        """

        with xact():
            # Gather objs we'll need save after
            old_m2ms = self._gather_m2ms()
            old_reverses = self._gather_reverses()

            for k, v in attrs.items():
                setattr(self, k, v)

            # Do the clone
            self.prep_for_clone()
            self.validate_unique()
            # Prevent last save from changing
            self.save(last_save=self.last_save)

            # save m2ms
            self._set_m2ms(old_m2ms)
            # Prevent last save from changing
            self.save(last_save=self.last_save)

            # save reverses
            self._clone_reverses(old_reverses)
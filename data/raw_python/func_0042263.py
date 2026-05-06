def _get_choices(self):
        """
        Redefine standard method.
        """
        if not self._choices:
            self._choices = tuple(
                (x.name, getattr(x, 'verbose_name', x.name) or x.name)
                for x in self.choices_class.constants()
            )
        return self._choices
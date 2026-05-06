def _set_thing(self, thing, value):
        """Convenience method for `_set_year`, `_set_month`..."""
        try:
            value = int(value)
        except (TypeError, ValueError):
            raise TypeError(
                f'Changing the {thing} of a `Date` instance is only '
                f'allowed via numbers, but the given value `{value}` '
                f'is of type `{type(value)}` instead.')
        kwargs = {}
        for unit in ('year', 'month', 'day', 'hour', 'minute', 'second'):
            kwargs[unit] = getattr(self, unit)
        kwargs[thing] = value
        self.datetime = datetime.datetime(**kwargs)
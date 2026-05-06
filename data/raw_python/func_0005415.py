def _get_date_tuple(self, mediafile):
        """Get a 3-item sequence representing the date consisting of a
        year, month, and day number. Each number is either an integer or
        None.
        """
        # Get the underlying data and split on hyphens and slashes.
        datestring = super(DateField, self).__get__(mediafile, None)
        if isinstance(datestring, six.string_types):
            datestring = re.sub(r'[Tt ].*$', '', six.text_type(datestring))
            items = re.split('[-/]', six.text_type(datestring))
        else:
            items = []

        # Ensure that we have exactly 3 components, possibly by
        # truncating or padding.
        items = items[:3]
        if len(items) < 3:
            items += [None] * (3 - len(items))

        # Use year field if year is missing.
        if not items[0] and hasattr(self, '_year_field'):
            items[0] = self._year_field.__get__(mediafile)

        # Convert each component to an integer if possible.
        items_ = []
        for item in items:
            try:
                items_.append(int(item))
            except (TypeError, ValueError):
                items_.append(None)
        return items_
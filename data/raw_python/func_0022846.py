def compute_positions(cls, screen_width, line):
        """Compute the relative position of the fields on a given line.

        Args:
            screen_width (int): the width of the screen
            line (mpdlcd.display_fields.Field list): the list of fields on the
                line

        Returns:
            ((int, mpdlcd.display_fields.Field) list): the positions of fields,
                as (position, field) tuples.

        Raises:
            FormatError: if the line contains more than one flexible field, or
                is too long for the screen size.
        """
        # First index
        left = 1
        # Last index
        right = screen_width + 1
        # Current 'flexible' field
        flexible = None

        # Compute the space to the left and to the right of the (optional)
        # flexible field.
        for field in line:
            if field.is_flexible():
                if flexible:
                    raise FormatError(
                        'There can be only one flexible field per line.')
                flexible = field

            elif not flexible:
                left += field.width

            else:
                # Met a 'flexible', computing from the right
                right -= field.width

        # Available space for the 'flexible' field
        available = right - left
        if available <= 0:
            raise FormatError("Too much data for screen width")

        if flexible:
            if available < 1:
                raise FormatError(
                    "Not enough space to display flexible field %s" %
                    flexible.name)

            flexible.width = available

        positions = []
        left = 1
        for field in line:
            positions.append((left, field))
            left += field.width

        logger.debug('Positions are %r', positions)
        return positions
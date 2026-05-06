def from_floats(red, green, blue):
        """Return a new Color object from red/green/blue values from 0.0 to 1.0."""

        return Color(int(red * Color.MAX_VALUE),
                     int(green * Color.MAX_VALUE),
                     int(blue * Color.MAX_VALUE))
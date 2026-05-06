def _to_star(self):
        """Save :class:`~nmrstarlib.nmrstarlib.StarFile` into NMR-STAR or CIF formatted string.

        :return: NMR-STAR string.
        :rtype: :py:class:`str`
        """
        star_str = io.StringIO()
        self.print_file(star_str)
        return star_str.getvalue()
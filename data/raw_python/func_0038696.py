def add_colors_from_file(self, system, f_or_filename):
        """Add color definition to a given color system.

        You may pass either a file-like object or a filename string pointing
        to a color definition csv file. Each line in that input file should
        look like this::

            café au lait,a67b5b

        i.e. a color name and a sRGB hex code, separated by by comma (``,``). Note that
        this is standard excel-style csv format without headers.

        You may add to already existing color system. Previously existing color
        definitions of the same (normalized) name will be overwritten,
        regardless of the color system.

        Args:
          system (string): The color system the colors should be added to
            (e.g. ``"en"``).
          color_definitions (filename, or file-like object): Either
            a filename, or a file-like object pointing to a color definition
            csv file (excel style).

        """
        if hasattr(f_or_filename, "read"):
            colors = (row for row in csv.reader(f_or_filename) if row)
        else:
            with open(f_or_filename, "rb") as f:
                colors = [row for row in csv.reader(f) if row]

        self.add_colors(system, colors)
def _build_casefold_map(self):
        """
        Function for parsing the case folding data from the Unicode Character
        Database (UCD) and generating a lookup table.  For more info on the UCD,
        see the following website: https://www.unicode.org/ucd/
        """
        self._casefold_map = defaultdict(dict)
        filename = "CaseFolding.txt"
        current_dir = os.path.abspath(os.path.dirname(__file__))
        with codecs.open(os.path.join(current_dir, filename), mode="r", encoding="utf-8") as fp:
            for line in fp:
                if not line.strip() or line.startswith("#"):
                    continue  # Skip empty lines or lines that are comments (comments start with '#')
                code, status, mapping, name = line.split(";")
                src = _hexstr_to_unichr(code)
                target = u"".join([_hexstr_to_unichr(c) for c in mapping.strip().split()])
                self._casefold_map[status.strip()][src] = target
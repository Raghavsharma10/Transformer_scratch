def _load_unicode_block_info(self):
        """
        Function for parsing the Unicode block info from the Unicode Character
        Database (UCD) and generating a lookup table.  For more info on the UCD,
        see the following website: https://www.unicode.org/ucd/
        """
        filename = "Blocks.txt"
        current_dir = os.path.abspath(os.path.dirname(__file__))
        with codecs.open(os.path.join(current_dir, filename), mode="r", encoding="utf-8") as fp:
            for line in fp:
                if not line.strip() or line.startswith("#"):
                    continue  # Skip empty lines or lines that are comments (comments start with '#')
                # Format: Start Code..End Code; Block Name
                block_range, block_name = line.strip().split(";")
                start_range, end_range = block_range.strip().split("..")
                self._unicode_blocks[six.moves.range(int(start_range, 16), int(end_range, 16) + 1)] = block_name.strip()
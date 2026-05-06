def blocks_by_text(cls, text):
        """Return a list of blocks matching the given :attr:`text`.

        Capitalisation and spaces are ignored.

        """
        text = kurt.BlockType._strip_text(text)
        matches = []
        for block in cls.blocks:
            for pbt in block.conversions:
                if pbt.stripped_text == text:
                    matches.append(block)
                    break
        return matches
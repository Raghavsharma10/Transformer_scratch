def _remove_trailing_spaces(text: str) -> str:
        """
        Remove trailing spaces and tabs

        :param text: Text to clean up
        :return:
        """
        clean_text = str()

        for line in text.splitlines(True):
            # remove trailing spaces (0x20) and tabs (0x09)
            clean_text += line.rstrip("\x09\x20")

        return clean_text
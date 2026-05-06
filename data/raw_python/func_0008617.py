def _strip_text(text):
        """Returns text with spaces and inserts removed."""
        text = re.sub(r'[ ,?:]|%s', "", text.lower())
        for chr in "-%":
            new_text = text.replace(chr, "")
            if new_text:
                text = new_text
        return text.lower()
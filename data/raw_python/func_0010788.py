def replace(cls, replacement_string):
        """Add in valid emojis in a string where a valid emoji is between ::"""
        e = cls()

        def _replace_emoji(match):
            val = match.group(1)
            if val in e:
                return e._image_string(match.group(1))
            else:
                return match.group(0)

        return e._pattern.sub(_replace_emoji, replacement_string)
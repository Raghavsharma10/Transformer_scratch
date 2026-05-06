def _merge(cls, default, user):
        """Append user options to default options. Return yala section."""
        section = cls._CFG_SECTION
        merged = default[section]

        if section not in user:
            return merged

        user = user[section]
        for key, value in user.items():
            if key in merged:
                merged[key] += ' ' + value
            else:
                merged[key] = value
        return merged
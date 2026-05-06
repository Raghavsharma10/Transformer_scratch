def html_diff(self, old, new):
        """
        Return HTML formatted character-based diff between old and new (used for CS50 IDE).
        """
        def html_transition(old_type, new_type):
            tags = []
            for tag in [("/", old_type), ("", new_type)]:
                if tag[1] not in ["+", "-"]:
                    continue
                tags.append("<{}{}>".format(tag[0], "ins" if tag[1] == "+" else "del"))
            return "".join(tags)

        return self._char_diff(old, new, html_transition, fmt=cgi.escape)
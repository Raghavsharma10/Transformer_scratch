def char_diff(self, old, new):
        """
        Return color-coded character-based diff between `old` and `new`.
        """
        def color_transition(old_type, new_type):
            new_color = termcolor.colored("", None, "on_red" if new_type ==
                                          "-" else "on_green" if new_type == "+" else None)
            return "{}{}".format(termcolor.RESET, new_color[:-len(termcolor.RESET)])

        return self._char_diff(old, new, color_transition)
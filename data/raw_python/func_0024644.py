def unified(old, new):
        """
        Returns a generator yielding a unified diff between `old` and `new`.
        """
        for diff in difflib.ndiff(old.splitlines(), new.splitlines()):
            if diff[0] == " ":
                yield diff
            elif diff[0] == "?":
                continue
            else:
                yield termcolor.colored(diff, "red" if diff[0] == "-" else "green", attrs=["bold"])
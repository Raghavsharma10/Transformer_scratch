def run_diff(self):
        """
        Run checks on self.files, printing diff of styled/unstyled output to stdout.
        """
        files = tuple(self.files)
        # Use same header as more.
        header, footer = (termcolor.colored("{0}\n{{}}\n{0}\n".format(
            ":" * 14), "cyan"), "\n") if len(files) > 1 else ("", "")

        for file in files:
            print(header.format(file), end="")
            try:
                results = self._check(file)
            except Error as e:
                termcolor.cprint(e.msg, "yellow", file=sys.stderr)
                continue

            # Display results
            if results.diffs:
                print()
                print(*self.diff(results.original, results.styled), sep="\n")
                print()
                conjunction = "And"
            else:
                termcolor.cprint("Looks good!", "green")
                conjunction = "But"

            if results.diffs:
                for type, c in sorted(self._warn_chars):
                    color, verb = ("on_green", "insert") if type == "+" else ("on_red", "delete")
                    termcolor.cprint(c, None, color, end="")
                    termcolor.cprint(" means that you should {} a {}.".format(
                        verb, "newline" if c == "\\n" else "tab"), "yellow")

            if results.comment_ratio < results.COMMENT_MIN:
                termcolor.cprint("{} consider adding more comments!".format(conjunction), "yellow")

            if (results.comment_ratio < results.COMMENT_MIN or self._warn_chars) and results.diffs:
                print()
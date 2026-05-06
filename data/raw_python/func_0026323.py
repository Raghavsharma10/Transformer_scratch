def format_text(self, include_password=True, use_colors=None, padding=True, filters=()):
        """
        Format :attr:`text` for viewing on a terminal.

        :param include_password: :data:`True` to include the password in the
                                 formatted text, :data:`False` to exclude the
                                 password from the formatted text.
        :param use_colors: :data:`True` to use ANSI escape sequences,
                           :data:`False` otherwise. When this is :data:`None`
                           :func:`~humanfriendly.terminal.terminal_supports_colors()`
                           will be used to detect whether ANSI escape sequences
                           are supported.
        :param padding: :data:`True` to add empty lines before and after the
                        entry and indent the entry's text with two spaces,
                        :data:`False` to skip the padding.
        :param filters: An iterable of regular expression patterns (defaults to
                        an empty tuple). If a line in the entry's text matches
                        one of these patterns it won't be shown on the
                        terminal.
        :returns: The formatted entry (a string).
        """
        # Determine whether we can use ANSI escape sequences.
        if use_colors is None:
            use_colors = terminal_supports_colors()
        # Extract the password (first line) from the entry.
        lines = self.text.splitlines()
        password = lines.pop(0).strip()
        # Compile the given patterns to case insensitive regular expressions
        # and use them to ignore lines that match any of the given filters.
        patterns = [coerce_pattern(f, re.IGNORECASE) for f in filters]
        lines = [l for l in lines if not any(p.search(l) for p in patterns)]
        text = trim_empty_lines("\n".join(lines))
        # Include the password in the formatted text?
        if include_password:
            text = "Password: %s\n%s" % (password, text)
        # Add the name to the entry (only when there's something to show).
        if text and not text.isspace():
            title = " / ".join(split(self.name, "/"))
            if use_colors:
                title = ansi_wrap(title, bold=True)
            text = "%s\n\n%s" % (title, text)
        # Highlight the entry's text using ANSI escape sequences.
        lines = []
        for line in text.splitlines():
            # Check for a "Key: Value" line.
            match = KEY_VALUE_PATTERN.match(line)
            if match:
                key = "%s:" % match.group(1).strip()
                value = match.group(2).strip()
                if use_colors:
                    # Highlight the key.
                    key = ansi_wrap(key, color=HIGHLIGHT_COLOR)
                    # Underline hyperlinks in the value.
                    tokens = value.split()
                    for i in range(len(tokens)):
                        if "://" in tokens[i]:
                            tokens[i] = ansi_wrap(tokens[i], underline=True)
                    # Replace the line with a highlighted version.
                    line = key + " " + " ".join(tokens)
            if padding:
                line = "  " + line
            lines.append(line)
        text = "\n".join(lines)
        text = trim_empty_lines(text)
        if text and padding:
            text = "\n%s\n" % text
        return text
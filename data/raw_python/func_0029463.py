def replace(self, src: str) -> str:
        """
        Extends LaTeX syntax via regex preprocess
        :param src: str
            LaTeX string
        :return: str
            New LaTeX string
        """
        if not self.readied:
            self.ready()

        # Brackets + simple pre replacements:
        src = self._dict_replace(self.simple_pre, src)

        # Superscripts and subscripts + pre regexps:
        for regex, replace in self.regex_pre:
            src = regex.sub(replace, src)

        # Unary and binary operators:
        src = self._operators_replace(src)

        # Loop regexps:
        src_prev = src
        for i in range(self.max_iter):
            for regex, replace in self.loop_regexps:
                src = regex.sub(replace, src)
            if src_prev == src:
                break
            else:
                src_prev = src

        # Post regexps:
        for regex, replace in self.regex_post:
            src = regex.sub(replace, src)

        # Simple post replacements:
        src = self._dict_replace(self.simple_post, src)

        # Escape characters:
        src = self.escapes_regex.sub(r'\1', src)

        return src
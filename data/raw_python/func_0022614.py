def _next_pattern(self):
        """Parses the next pattern by matching each in turn."""
        current_state = self.state_stack[-1]
        position = self._position
        for pattern in self.patterns:
            if current_state not in pattern.states:
                continue

            m = pattern.regex.match(self.source, position)
            if not m:
                continue

            position = m.end()
            token = None

            if pattern.next_state:
                self.state_stack.append(pattern.next_state)

            if pattern.action:
                callback = getattr(self, pattern.action, None)
                if callback is None:
                    raise RuntimeError(
                        "No method defined for pattern action %s!" %
                        pattern.action)

                if "token" in m.groups():
                    value = m.group("token")
                else:
                    value = m.group(0)
                token = callback(string=value, match=m,
                                 pattern=pattern)

            self._position = position

            return token

        self._error("Don't know how to match next. Did you forget quotes?",
                    start=self._position, end=self._position + 1)
def _CheckLine(self, line):
    """Passes the line through each rule until a match is made.

    Args:
      line: A string, the current input line.
    """
    for rule in self._cur_state:
      matched = self._CheckRule(rule, line)
      if matched:
        for value in matched.groupdict():
          self._AssignVar(matched, value)

        if self._Operations(rule):
          # Not a Continue so check for state transition.
          if rule.new_state:
            if rule.new_state not in ('End', 'EOF'):
              self._cur_state = self.states[rule.new_state]
            self._cur_state_name = rule.new_state
          break
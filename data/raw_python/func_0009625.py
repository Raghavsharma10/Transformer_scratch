def run(self, text):
    '''Run lexer rules against a source text

    Args:
      text (str): Text to apply lexer to

    Yields:
      A sequence of lexer matches.
    '''

    stack = ['root']
    pos = 0

    patterns = self.tokens[stack[-1]]

    while True:
      for pat, action, new_state in patterns:
        m = pat.match(text, pos)
        if m:
          if action:
            #print('## MATCH: {} -> {}'.format(m.group(), action))
            yield (pos, m.end()-1), action, m.groups()

          pos = m.end()

          if new_state:
            if isinstance(new_state, int): # Pop states
              del stack[new_state:]
            else:
              stack.append(new_state)

            #print('## CHANGE STATE:', pos, new_state, stack)
            patterns = self.tokens[stack[-1]]

          break

      else:
        try:
          if text[pos] == '\n':
            pos += 1
            continue
          pos += 1
        except IndexError:
          break
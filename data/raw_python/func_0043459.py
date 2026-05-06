def compile(pattern, flags=0, sep=None, split_prefix=False):
  '''
  Converts a glob-matching pattern (using Apache Cocoon style rules)
  to a regular expression, which basically means that the following
  characters have special meanings:

  * ``?``:     matches any single character excluding the separator character
  * ``*``:     matches zero or more characters excluding the separator character
  * ``**``:    matches zero or more characters including the separator character
  * ``\``:     escape character used to precede any of the others for a literal
  * ``[...]``: matches any character in the specified regex-style range
  * ``{...}``: inlines a regex expression

  :Parameters:

  sep : str; default: "/"

    The `sep` parameter specifies the hierarchical path component
    separator to use. By default, it uses the unix-style forward-slash
    separator (``"/"``), but can be overriden to be a sequence of
    alternative valid hierarchical path component separator characters.
    Note that although `sep` *could* be set to both forward- and back-
    slashes (i.e. ``"/\\"``) to, theoretically, support either unix- and
    windows-style path components, this has the significant flaw that
    then *both* characters can be used within the same path as
    separators.

  flags : int; default: 0

    The `flags` bit mask can contain all the standard `re` flags, in
    addition to the ``globre.EXACT`` flag. If EXACT is set, then the
    returned regex will include a leading '^' and trailing '$', meaning
    that the regex must match the entire string, from beginning to end.

  split_prefix : bool; default: false

    If `split_prefix` is truthy, the return value becomes a tuple with
    the first element set to any initial non-wildcarded string found in
    the pattern. The second element remains the regex object as before.
    For example, the pattern ``foo/**.ini`` would result in a tuple
    equivalent to ``('foo/', re.compile('foo/.*\\.ini'))``.
  '''

  prefix = None
  expr   = ''

  if sep is None:
    sep = '/'
  if not sep:
    TypeError('invalid parameter "sep" value: %r' % (sep,))
  if set(sep) & set(SPECIAL_CHARS):
    TypeError('parameter "sep" cannot contain any of %r' % (SPECIAL_CHARS,))
  if len(sep) == 1:
    literal = re.escape
  else:
    def make_literal(sep):
      sep = '[' + re.escape(sep) + ']'
      sepcre = re.compile(sep)
      def _literal(text):
        return sep.join(sepcre.split(text))
      return _literal
    literal = make_literal(sep)
  if sep != '/':
    sep = re.escape(sep)

  for token in Tokenizer(pattern).tokens():
    if split_prefix and expr == '':
      prefix = token[1] if token[0] == Tokenizer.LITERAL else ''
    if token[0] == Tokenizer.LITERAL:
      expr += literal(token[1])
    elif token[0] == Tokenizer.SINGLE:
      expr += '[^' + sep + ']'
    elif token[0] == Tokenizer.MULTIPLE:
      expr += '[^' + sep + ']*?'
    elif token[0] == Tokenizer.ANY:
      expr += '.*?'
    elif token[0] == Tokenizer.RANGE:
      expr += '[' + token[1] + ']'
    elif token[0] == Tokenizer.REGEX:
      expr += token[1]
    else:
      ValueError('unexpected token %r from globre.Tokenizer for glob: %s'
                 % (token, pattern))

  if flags & EXACT:
    if not expr.startswith('^'):
      expr = '^' + expr
    # todo: technically, the last "$" *could* be escaped and therefore
    #       an extra "$" would need to be added... but that is very unlikely.
    if not expr.endswith('$'):
      expr += '$'

  expr = re.compile(expr, flags=flags & ~ EXACT)

  if prefix is not None:
    return (prefix, expr)
  return expr
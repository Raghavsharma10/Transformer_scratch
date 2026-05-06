def make_grammar(allow_errors):
  """Make the part of the grammar that depends on whether we swallow errors or not."""
  if allow_errors in GRAMMAR_CACHE:
    return GRAMMAR_CACHE[allow_errors]

  tuple = p.Forward()
  catch_errors = p.Forward()
  catch_errors << (p.Regex('[^{};]*') - p.Optional(tuple) - p.Regex('[^;}]*'))

  def swallow_remainder():
    if allow_errors:
      return pattern('swallow_remainder', p.Suppress(catch_errors))
    return p.Empty()

  def swallow_errors(rule):
    """Extend the production rule by potentially eating errors.

    This does not return a p.NoMatch() because that messes up the error messages.
    """
    ret = rule
    if allow_errors:
      # Synchronize on the first semicolon or the first unbalanced closing curly
      ret = rule | pattern('catch_errors', parseWithLocation(p.Suppress(catch_errors), UnparseableNode))
    return ret

  class Grammar:
    keywords = ['and', 'or', 'not', 'if', 'then', 'else', 'include', 'inherit', 'null', 'true', 'false',
        'for', 'in']

    # This is a hack: this condition helps uselessly recursing into the grammar for
    # juxtapositions.
    early_abort_scan = ~p.oneOf([';', ',', ']', '}', 'for' ])

    expression = pattern('expression', p.Forward())

    comment = p.Regex('#') + ~p.FollowedBy(sym('.')) + p.restOfLine
    doc_comment = pattern('doc_comment', (sym('#.') - p.restOfLine))

    quotedIdentifier = pattern('quotedIdentifier', p.QuotedString('`', multiline=False))

    # - Must start with an alphascore
    # - May contain alphanumericscores and special characters such as : and -
    # - Must not end in a special character
    identifier = pattern('identifier', parseWithLocation(quotedIdentifier | p.Regex(r'[a-zA-Z_]([a-zA-Z0-9_:-]*[a-zA-Z0-9_])?'), Identifier))

    # Variable identifier (can't be any of the keywords, which may have lower matching priority)
    variable = pattern('variable', ~p.MatchFirst(p.oneOf(keywords)) + pattern('identifier', parseWithLocation(identifier.copy(), Var)))

    # Contants
    integer = pattern('integer', parseWithLocation(p.Word(p.nums), convertAndMake(int, Literal)))
    floating = pattern('floating', parseWithLocation(p.Regex(r'\d*\.\d+'), convertAndMake(float, Literal)))
    dq_string = pattern('dq_string', parseWithLocation(p.QuotedString('"', escChar='\\', unquoteResults=False, multiline=True), convertAndMake(unquote, Literal)))
    sq_string = pattern('sq_string', parseWithLocation(p.QuotedString("'", escChar='\\', unquoteResults=False, multiline=True), convertAndMake(unquote, Literal)))
    boolean = pattern('boolean', parseWithLocation(p.Keyword('true') | p.Keyword('false'), convertAndMake(mkBool, Literal)))
    null = pattern('null', parseWithLocation(p.Keyword('null'), Null))

    # List
    list_ = pattern('list', parseWithLocation(bracketedList('[', ']', ',', expression), List))

    # Tuple
    inherit = pattern('inherit', (kw('inherit') - p.ZeroOrMore(variable)).setParseAction(inheritNodes))
    schema_spec = pattern('schema_spec', parseWithLocation(p.Optional(p.Keyword('private').setParseAction(lambda: True), default=False)
                  - p.Optional(p.Keyword('required').setParseAction(lambda: True), default=False)
                  - p.Optional(expression, default=any_schema_expr), MemberSchemaNode))
    optional_schema = pattern('optional_schema', p.Optional(p.Suppress(':') - schema_spec, default=no_schema))

    expression_value = pattern('expression_value', sym('=') - swallow_errors(expression))
    void_value = pattern('void_value', parseWithLocation(p.FollowedBy(sym(';') | sym('}')), lambda loc: Void(loc, 'nonameyet')))
    member_value = pattern('member_value', swallow_errors(expression_value | void_value))
    named_member = pattern('named_member', parseWithLocation(identifier - optional_schema - member_value - swallow_remainder(), TupleMemberNode))
    documented_member = pattern('documented_member', parseWithLocation(parseWithLocation(p.ZeroOrMore(doc_comment), DocComment) + named_member, attach_doc_comment))
    tuple_member = early_abort_scan + pattern('tuple_member', swallow_errors(inherit | documented_member) - swallow_remainder())

    ErrorAwareTupleNode = functools.partial(TupleNode, allow_errors)
    tuple_members = pattern('tuple_members', parseWithLocation(listMembers(';', tuple_member), ErrorAwareTupleNode))
    tuple << pattern('tuple', parseWithLocation(bracketedList('{', '}', ';', tuple_member, allow_missing_close=allow_errors), ErrorAwareTupleNode))

    # Argument list will live by itself as a atom. Actually, it's a tuple, but we
    # don't call it that because we use that term for something else already :)
    arg_list = pattern('arg_list', bracketedList('(', ')', ',', expression).setParseAction(ArgList))

    parenthesized_expr = pattern('parenthesized_expr', (sym('(') - expression - ')').setParseAction(head))

    unary_op = pattern('unary_op', (p.oneOf(' '.join(functions.unary_operators.keys())) - expression).setParseAction(mkUnOp))

    if_then_else = pattern('if_then_else', parseWithLocation(kw('if') + expression +
                    kw('then') + expression +
                    kw('else') + expression, Condition))

    list_comprehension = pattern('list_comprehension', parseWithLocation(sym('[') + expression + kw('for') + variable + kw('in') +
        expression + p.Optional(kw('if') + expression) + sym(']'), ListComprehension))


    # We don't allow space-application here
    # Now our grammar is becoming very dirty and hackish
    deref = pattern('deref', p.Forward())
    include = pattern('include', parseWithLocation(kw('include') - deref, Include))

    atom = pattern('atom', (tuple
            | sq_string
            | dq_string
            | variable
            | floating
            | integer
            | boolean
            | list_
            | null
            | unary_op
            | parenthesized_expr
            | if_then_else
            | include
            | list_comprehension
            ))

    # We have two different forms of function application, so they can have 2
    # different precedences. This one: fn(args), which binds stronger than
    # dereferencing (fn(args).attr == (fn(args)).attr)
    applic1 = pattern('applic1', parseWithLocation(atom - p.ZeroOrMore(arg_list), mkApplications))

    # Dereferencing of an expression (obj.bar)
    deref << parseWithLocation(applic1 - p.ZeroOrMore(p.Suppress('.') - swallow_errors(identifier)), mkDerefs)

    # All binary operators at various precedence levels go here:
    # This piece of code does the moral equivalent of:
    #
    #     T = F*F | F/F | F
    #     E = T+T | T-T | T
    #
    # etc.
    term = deref
    for op_level in functions.binary_operators_before_juxtaposition:
      operator_syms = list(op_level.keys())
      term = (term - p.ZeroOrMore(p.oneOf(operator_syms) - term)).setParseAction(mkBinOps)

    # Juxtaposition function application (fn arg), must be 1-arg every time
    applic2 = pattern('applic2', parseWithLocation(term - p.ZeroOrMore(early_abort_scan + term), mkApplications))

    term = applic2
    for op_level in functions.binary_operators_after_juxtaposition:
      operator_syms = list(op_level.keys())
      term = (term - p.ZeroOrMore(p.oneOf(operator_syms) - term)).setParseAction(mkBinOps)

    expression << term

    # Two entry points: start at an arbitrary expression, or expect the top-level
    # scope to be a tuple.
    start = pattern('start', expression.copy().ignore(comment))
    start_tuple = tuple_members.ignore(comment)
  GRAMMAR_CACHE[allow_errors] = Grammar
  return Grammar
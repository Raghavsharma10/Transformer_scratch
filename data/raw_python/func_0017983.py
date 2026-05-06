def reads(s, filename, loader, implicit_tuple, allow_errors):
  """Load but don't evaluate a GCL expression from a string."""
  try:
    the_context.filename = filename
    the_context.loader = loader

    grammar = make_grammar(allow_errors=allow_errors)
    root = grammar.start_tuple if implicit_tuple else grammar.start

    return root.parseWithTabs().parseString(s, parseAll=True)[0]
  except (p.ParseException, p.ParseSyntaxException) as e:
    loc = SourceLocation(s, find_offset(s, e.lineno, e.col))
    raise exceptions.ParseError(the_context.filename, loc, e.msg)
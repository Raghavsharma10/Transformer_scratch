def inflate_context_tuple(ast_rootpath, root_env):
  """Instantiate a Tuple from a TupleNode.

  Walking the AST tree upwards, evaluate from the root down again.
  """
  with util.LogTime('inflate_context_tuple'):
    # We only need to look at tuple members going down.
    inflated = ast_rootpath[0].eval(root_env)
    current = inflated
    env = root_env
    try:
      for node in ast_rootpath[1:]:
        if is_tuple_member_node(node):
          assert framework.is_tuple(current)
          with util.LogTime('into tuple'):
            thunk, env = inflated.get_thunk_env(node.name)
            current = framework.eval(thunk, env)

        elif framework.is_list(current):
          with util.LogTime('eval thing'):
            current = framework.eval(node, env)

        if framework.is_tuple(current):
          inflated = current
    except (gcl.EvaluationError, ast.UnparseableAccess):
      # Eat evaluation error, probably means the rightmost tuplemember wasn't complete.
      # Return what we have so far.
      pass
    return inflated
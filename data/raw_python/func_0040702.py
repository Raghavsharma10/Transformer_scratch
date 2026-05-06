def canonicalize_spec(spec, parent_context):
  """Push all context declarations to the leaves of a nested test specification."""

  test_specs = {k:v for (k,v) in spec.items() if k.startswith("Test")}
  local_context = {k:v for (k,v) in spec.items() if not k.startswith("Test")}

  context = reduce_contexts(parent_context, local_context)

  if test_specs:
    return {k: canonicalize_spec(v, context) for (k,v) in test_specs.items()}
  else:
    program_chunks = sum([resolve_module(m,context['Definitions']) for m in context['Modules']],[]) + [context['Program']]
    test_spec = {
      'Arguments': context['Arguments'],
      'Program': "\n".join(program_chunks),
      'Expect': context['Expect'],
    }
    return test_spec
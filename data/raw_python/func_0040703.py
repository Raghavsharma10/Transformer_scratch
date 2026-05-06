def flatten_spec(spec, prefix,joiner=" :: "):
  """Flatten a canonical specification with nesting into one without nesting.
  When building unique names, concatenate the given prefix to the local test
  name without the "Test " tag."""

  if any(filter(operator.methodcaller("startswith","Test"),spec.keys())):
    flat_spec = {}
    for (k,v) in spec.items():
      flat_spec.update(flatten_spec(v,prefix + joiner + k[5:]))
    return flat_spec 
  else:
    return {"Test "+prefix: spec}
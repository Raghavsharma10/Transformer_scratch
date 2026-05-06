def _make_tuple(self, env):
    """Instantiate the Tuple based on this TupleNode."""
    t = runtime.Tuple(self, env, dict2tuple)
    # A tuple also provides its own schema spec
    schema = schema_spec_from_tuple(t)
    t.attach_schema(schema)
    return t
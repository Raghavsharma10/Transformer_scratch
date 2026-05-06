def eval(thunk, env):
  """Evaluate a thunk in an environment.

  Will defer the actual evaluation to the thunk itself, but adds two things:
  caching and recursion detection.

  Since we have to use a global evaluation stack (because there is a variety of functions that may
  be invoked, not just eval() but also __getitem__, and not all of them can pass along a context
  object), GCL evaluation is not thread safe.

  With regard to schemas:

  - A schema can be passed in from outside. The returned object will be validated to see that it
    conforms to the schema. The schema will be attached to the value if possible.
  - Some objects may contain their own schema, such as tuples. This would be out of scope of the
    eval() function, were it not for:
  - Schema validation can be disabled in an evaluation call stack. This is useful if we're
    evaluating a tuple only for its schema information. At that point, we're not interested if the
    object is value-complete.
  """
  key = Activation.key(thunk, env)
  if Activation.activated(key):
    raise exceptions.RecursionError('Reference cycle')

  with Activation(key):
    return eval_cache.get(key, thunk.eval, env)
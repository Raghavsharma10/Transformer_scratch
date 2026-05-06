def expand(data):
  '''Generates configuration sets based on the YAML input contents

  For an introduction to the YAML mark-up, just search the net. Here is one of
  its references: https://en.wikipedia.org/wiki/YAML

  A configuration set corresponds to settings for **all** variables in the
  input template that needs replacing. For example, if your template mentions
  the variables ``name`` and ``version``, then each configuration set should
  yield values for both ``name`` and ``version``.

  For example:

  .. code-block:: yaml

     name: [john, lisa]
     version: [v1, v2]


  This should yield to the following configuration sets:

  .. code-block:: python

     [
       {'name': 'john', 'version': 'v1'},
       {'name': 'john', 'version': 'v2'},
       {'name': 'lisa', 'version': 'v1'},
       {'name': 'lisa', 'version': 'v2'},
     ]


  Each key in the input file should correspond to either an object or a YAML
  array. If the object is a list, then we'll iterate over it for every possible
  combination of elements in the lists. If the element in question is not a
  list, then it is considered unique and repeated for each yielded
  configuration set. Example

  .. code-block:: yaml

     name: [john, lisa]
     version: [v1, v2]
     text: >
        hello,
        world!

  Should yield to the following configuration sets:

  .. code-block:: python

     [
       {'name': 'john', 'version': 'v1', 'text': 'hello, world!'},
       {'name': 'john', 'version': 'v2', 'text': 'hello, world!'},
       {'name': 'lisa', 'version': 'v1', 'text': 'hello, world!'},
       {'name': 'lisa', 'version': 'v2', 'text': 'hello, world!'},
     ]

  Keys starting with one `_` (underscore) are treated as "unique" objects as
  well. Example:

  .. code-block:: yaml

     name: [john, lisa]
     version: [v1, v2]
     _unique: [i1, i2]

  Should yield to the following configuration sets:

  .. code-block:: python

     [
       {'name': 'john', 'version': 'v1', '_unique': ['i1', 'i2']},
       {'name': 'john', 'version': 'v2', '_unique': ['i1', 'i2']},
       {'name': 'lisa', 'version': 'v1', '_unique': ['i1', 'i2']},
       {'name': 'lisa', 'version': 'v2', '_unique': ['i1', 'i2']},
     ]


  Parameters:

    data (str): YAML data to be parsed


  Yields:

    dict: A dictionary of key-value pairs for building the templates

  '''

  data = _ordered_load(data, yaml.SafeLoader)

  # separates "unique" objects from the ones we have to iterate
  # pre-assemble return dictionary
  iterables = dict()
  unique = dict()
  for key, value in data.items():
    if isinstance(value, list) and not key.startswith('_'):
      iterables[key] = value
    else:
      unique[key] = value

  # generates all possible combinations of iterables
  for values in itertools.product(*iterables.values()):
    retval = dict(unique)
    keys = list(iterables.keys())
    retval.update(dict(zip(keys, values)))
    yield retval
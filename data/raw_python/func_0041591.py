def generate(variables, template):
  '''Yields a resolved "template" for each config set and dumps on output

  This function will extrapolate the ``template`` file using the contents of
  ``variables`` and will output individual (extrapolated, expanded) files in
  the output directory ``output``.


  Parameters:

    variables (str): A string stream containing the variables to parse, in YAML
      format as explained on :py:func:`expand`.

    template (str): A string stream containing the template to extrapolate


  Yields:

    str: A generated template you can save


  Raises:

    jinja2.UndefinedError: if a variable used in the template is undefined

  '''

  env = jinja2.Environment(undefined=jinja2.StrictUndefined)
  for c in expand(variables):
    c['rc'] = rc
    yield env.from_string(template).render(c)
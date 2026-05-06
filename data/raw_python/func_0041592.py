def aggregate(variables, template):
  '''Generates a resolved "template" for **all** config sets and returns

  This function will extrapolate the ``template`` file using the contents of
  ``variables`` and will output a single (extrapolated, expanded) file.


  Parameters:

    variables (str): A string stream containing the variables to parse, in YAML
      format as explained on :py:func:`expand`.

    template (str): A string stream containing the template to extrapolate


  Returns:

    str: A generated template you can save


  Raises:

    jinja2.UndefinedError: if a variable used in the template is undefined

  '''

  env = jinja2.Environment(undefined=jinja2.StrictUndefined)
  d = {'cfgset': list(expand(variables)), 'rc': rc}
  return env.from_string(template).render(d)
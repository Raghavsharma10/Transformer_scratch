def harvest_all_openaire_projects():
    """Reharvest all grants from OpenAIRE.

    Harvest all OpenAIRE grants in a chain to prevent OpenAIRE
    overloading from multiple parallel harvesting.
    """
    setspecs = current_app.config['OPENAIRE_GRANTS_SPECS']
    chain(harvest_openaire_projects.s(setspec=setspec)
          for setspec in setspecs).apply_async()
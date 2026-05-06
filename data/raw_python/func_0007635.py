def describe(profile, description):
    """
    Generate a query by describing it as a series of actions
    and parameters to those actions. These map directly
    to Query methods and arguments to those methods.

    This is an alternative to the chaining interface.
    Mostly useful if you'd like to put your queries
    in a file, rather than in Python code.
    """
    api_type = description.pop('type', 'core')
    api = getattr(profile, api_type)
    return refine(api.query, description)
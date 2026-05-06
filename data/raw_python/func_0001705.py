def get_long_description():
    """ Retrieve the long description from DESCRIPTION.rst """
    here = os.path.abspath(os.path.dirname(__file__))

    with copen(os.path.join(here, 'DESCRIPTION.rst'), encoding='utf-8') as description:
        return description.read()
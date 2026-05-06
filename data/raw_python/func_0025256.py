def pypirc_temp(index_url):
    """ Create a temporary pypirc file for interaction with twine """
    pypirc_file = tempfile.NamedTemporaryFile(suffix='.pypirc', delete=False)
    print(pypirc_file.name)
    with open(pypirc_file.name, 'w') as fh:
        fh.write(PYPIRC_TEMPLATE.format(index_name=PYPIRC_TEMP_INDEX_NAME, index_url=index_url))
    return pypirc_file.name
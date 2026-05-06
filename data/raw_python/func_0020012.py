def main():
    """Sanitizes the loaded *.ipynb."""
    with open(sys.argv[1], 'r') as nbfile:
        notebook = json.load(nbfile)

    # remove kernelspec (venvs)
    try:
        del notebook['metadata']['kernelspec']
    except KeyError:
        pass

    # remove outputs and metadata, set execution counts to None
    for cell in notebook['cells']:
        try:
            if cell['cell_type'] == 'code':
                cell['outputs'] = []
                cell['execution_count'] = None
            cell['metadata'] = {}
        except KeyError:
            pass

    with open(sys.argv[1], 'w') as nbfile:
        json.dump(notebook, nbfile, indent=1)
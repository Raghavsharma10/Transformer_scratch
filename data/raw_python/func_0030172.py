def doc_parser():
    """Utility function to allow getting the arguments for a single command, for Sphinx documentation"""

    parser = argparse.ArgumentParser(
        prog='ambry',
        description='Ambry {}. Management interface for ambry, libraries '
                    'and repositories. '.format(ambry._meta.__version__))

    return parser
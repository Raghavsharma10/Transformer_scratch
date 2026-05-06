def setup(app):
    """Allow this module to be used as sphinx extension.
    This attaches the Sphinx hooks.

    :type app: sphinx.application.Sphinx
    """
    import sphinxcontrib_django.docstrings
    import sphinxcontrib_django.roles

    # Setup both modules at once. They can also be separately imported to
    # use only fragments of this package.
    sphinxcontrib_django.docstrings.setup(app)
    sphinxcontrib_django.roles.setup(app)
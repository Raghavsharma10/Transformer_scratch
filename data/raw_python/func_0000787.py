def setup(app):
    """Allow this package to be used as Sphinx extension.
    This is also called from the top-level ``__init__.py``.

    :type app: sphinx.application.Sphinx
    """
    from .patches import patch_django_for_autodoc

    # When running, make sure Django doesn't execute querysets
    patch_django_for_autodoc()

    # Generate docstrings for Django model fields
    # Register the docstring processor with sphinx
    app.connect('autodoc-process-docstring', improve_model_docstring)

    # influence skip rules
    app.connect("autodoc-skip-member", autodoc_skip)
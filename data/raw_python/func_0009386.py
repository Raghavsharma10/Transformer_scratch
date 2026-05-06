def tree(c):
    """
    Display documentation contents with the 'tree' program.
    """
    ignore = ".git|*.pyc|*.swp|dist|*.egg-info|_static|_build|_templates"
    c.run('tree -Ca -I "{0}" {1}'.format(ignore, c.sphinx.source))
def rotate_backups(directory, rotation_scheme, **options):
    """
    Rotate the backups in a directory according to a flexible rotation scheme.

    .. note:: This function exists to preserve backwards compatibility with
              older versions of the `rotate-backups` package where all of the
              logic was exposed as a single function. Please refer to the
              documentation of the :class:`RotateBackups` initializer and the
              :func:`~RotateBackups.rotate_backups()` method for an explanation
              of this function's parameters.
    """
    program = RotateBackups(rotation_scheme=rotation_scheme, **options)
    program.rotate_backups(directory)
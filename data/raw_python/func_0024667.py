def listdir(search_base, followlinks=False, filter='*',
            relpath=False, bestprefix=False, system=NIST):
    """This is a generator which recurses the directory tree
`search_base`, yielding 2-tuples of:

* The absolute/relative path to a discovered file
* A bitmath instance representing the "apparent size" of the file.

    - `search_base` - The directory to begin walking down.
    - `followlinks` - Whether or not to follow symbolic links to directories
    - `filter` - A glob (see :py:mod:`fnmatch`) to filter results with
      (default: ``*``, everything)
    - `relpath` - ``True`` to return the relative path from `pwd` or
      ``False`` (default) to return the fully qualified path
    - ``bestprefix`` - set to ``False`` to get ``bitmath.Byte``
      instances back instead.
    - `system` - Provide a preferred unit system by setting `system`
      to either ``bitmath.NIST`` (default) or ``bitmath.SI``.

.. note:: This function does NOT return tuples for directory entities.

.. note:: Symlinks to **files** are followed automatically

    """
    for root, dirs, files in os.walk(search_base, followlinks=followlinks):
        for name in fnmatch.filter(files, filter):
            _path = os.path.join(root, name)
            if relpath:
                # RELATIVE path
                _return_path = os.path.relpath(_path, '.')
            else:
                # REAL path
                _return_path = os.path.realpath(_path)

            if followlinks:
                yield (_return_path, getsize(_path, bestprefix=bestprefix, system=system))
            else:
                if os.path.isdir(_path) or os.path.islink(_path):
                    pass
                else:
                    yield (_return_path, getsize(_path, bestprefix=bestprefix, system=system))
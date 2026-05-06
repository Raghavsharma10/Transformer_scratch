def incfile(fname, fpointer, lrange=None, sdir=None):
    r"""
    Return a Python source file formatted in reStructuredText.

    .. role:: bash(code)
        :language: bash

    :param fname: File name, relative to environment variable
                  :bash:`PKG_DOC_DIR`
    :type  fname: string

    :param fpointer: Output function pointer. Normally is :code:`cog.out` but
                      other functions can be used for debugging
    :type  fpointer: function object

    :param lrange: Line range to include, similar to Sphinx
                   `literalinclude <http://www.sphinx-doc.org/en/master/usage
                   /restructuredtext/directives.html
                   #directive-literalinclude>`_ directive
    :type  lrange: string

    :param sdir: Source file directory. If None the :bash:`PKG_DOC_DIR`
                 environment variable is used if it is defined, otherwise
                 the directory where the module is located is used
    :type  sdir: string

    For example:

    .. code-block:: python

        def func():
            \"\"\"
            This is a docstring. This file shows how to use it:

            .. =[=cog
            .. import docs.support.incfile
            .. docs.support.incfile.incfile('func_example.py', cog.out)
            .. =]=
            .. code-block:: python

                # func_example.py
                if __name__ == '__main__':
                    func()

            .. =[=end=]=
            \"\"\"
            return 'This is func output'
    """
    # pylint: disable=R0914
    # Read file
    file_dir = (
        sdir
        if sdir
        else os.environ.get("PKG_DOC_DIR", os.path.abspath(os.path.dirname(__file__)))
    )
    fname = os.path.join(file_dir, fname)
    with open(fname, "r") as fobj:
        lines = fobj.readlines()
    # Eliminate spurious carriage returns in Microsoft Windows
    lines = [_homogenize_linesep(line) for line in lines]
    # Parse line specification
    inc_lines = (
        _proc_token(lrange, len(lines)) if lrange else list(range(1, len(lines) + 1))
    )
    # Produce output
    fpointer(".. code-block:: python" + os.linesep)
    fpointer(os.linesep)
    for num, line in enumerate(lines):
        if num + 1 in inc_lines:
            fpointer(
                "    " + line.replace("\t", "    ").rstrip() + os.linesep
                if line.strip()
                else os.linesep
            )
    fpointer(os.linesep)
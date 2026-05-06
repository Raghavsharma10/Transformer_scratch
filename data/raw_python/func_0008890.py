def make_long_description(marker=None, intro=None):
    """
    click_ is a framework to simplify writing composable commands for
    command-line tools. This package extends the click_ functionality
    by adding support for commands that use configuration files.

    .. _click: https://click.pocoo.org/

    EXAMPLE:

    A configuration file, like:

    .. code-block:: INI

        # -- FILE: foo.ini
        [foo]
        flag = yes
        name = Alice and Bob
        numbers = 1 4 9 16 25
        filenames = foo/xxx.txt
            bar/baz/zzz.txt

        [person.alice]
        name = Alice
        birthyear = 1995

        [person.bob]
        name = Bob
        birthyear = 2001

    can be processed with:

    .. code-block:: python

        # EXAMPLE:
    """
    if intro is None:
        intro = inspect.getdoc(make_long_description)

    with open("README.rst", "r") as infile:
        line = infile.readline()
        while not line.strip().startswith(marker):
            line = infile.readline()

        # -- COLLECT REMAINING: Usage example
        contents = infile.read()

    text = intro +"\n" + contents
    return text
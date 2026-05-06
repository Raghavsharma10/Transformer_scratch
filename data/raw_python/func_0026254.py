def autodoc_process_docstring(app, what, name, obj, options, lines):
    """Handler for the event emitted when autodoc processes a docstring.
    See http://sphinx-doc.org/ext/autodoc.html#event-autodoc-process-docstring.

    The TL;DR is that we can modify ``lines`` in-place to influence the output.
    """
    # check that only symbols that can be directly imported from ``callee``
    # package are being documented
    _, symbol = name.rsplit('.', 1)
    if symbol not in callee.__all__:
        raise SphinxError(
            "autodoc'd '%s' is not a part of the public API!" % name)

    # for classes exempt from automatic merging of class & __init__ docs,
    # pretend their __init__ methods have no docstring at all,
    # so that nothing will be appended to the class's docstring
    if what == 'class' and name in autoclass_content_exceptions:
        # amusingly, when autodoc reads the constructor's docstring
        # for appending it to class docstring, it will report ``what``
        # as 'class' (again!); hence we must check what it actually read
        ctor_docstring_lines = prepare_docstring(obj.__init__.__doc__)
        if lines == ctor_docstring_lines:
            lines[:] = []
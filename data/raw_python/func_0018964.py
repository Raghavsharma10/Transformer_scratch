def view(data, enc=None, start_pos=None, delimiter=None, hdr_rows=None,
         idx_cols=None, sheet_index=0, transpose=False, wait=None,
         recycle=None, detach=None, metavar=None, title=None):
    """View the supplied data in an interactive, graphical table widget.

    data: When a valid path or IO object, read it as a tabular text file. When
          a valid URI, a Blaze object is constructed and visualized. Any other
          supported datatype is visualized directly and incrementally *without
          copying*.

    enc: File encoding (such as "utf-8", normally autodetected).

    delimiter: Text file delimiter (normally autodetected).

    hdr_rows: For files or lists of lists, specify the number of header rows.
              For files only, a default of one header line is assumed.

    idx_cols: For files or lists of lists, specify the number of index columns.
              By default, no index is assumed.

    sheet_index: For multi-table files (such as xls[x]), specify the sheet
                 index to read, starting from 0. Defaults to the first.

    start_pos: A tuple of the form (y, x) specifying the initial cursor
               position. Negative offsets count from the end of the dataset.

    transpose: Transpose the resulting view.

    metavar: name of the variable being shown for display purposes (inferred
             automatically when possible).

    title: title of the data window.

    wait: Wait for the user to close the view before returning. By default, try
          to match the behavior of ``matplotlib.is_interactive()``. If
          matplotlib is not loaded, wait only if ``detach`` is also False. The
          default value can also be set through ``gtabview.WAIT``.

    recycle: Recycle the previous window instead of creating a new one. The
             default is True, and can also be set through ``gtabview.RECYCLE``.

    detach: Create a fully detached GUI thread for interactive use (note: this
            is *not* necessary if matplotlib is loaded). The default is False,
            and can also be set through ``gtabview.DETACH``.
    """
    global WAIT, RECYCLE, DETACH, VIEW

    model = read_model(data, enc=enc, delimiter=delimiter, hdr_rows=hdr_rows,
                       idx_cols=idx_cols, sheet_index=sheet_index,
                       transpose=transpose)
    if model is None:
        warnings.warn("cannot visualize the supplied data type: {}".format(type(data)),
                      category=RuntimeWarning)
        return None

    # setup defaults
    if wait is None: wait = WAIT
    if recycle is None: recycle = RECYCLE
    if detach is None: detach = DETACH
    if wait is None:
        if 'matplotlib' not in sys.modules:
            wait = not bool(detach)
        else:
            import matplotlib.pyplot as plt
            wait = not plt.isinteractive()

    # try to fetch the variable name in the upper stack
    if metavar is None:
        if isinstance(data, basestring):
            metavar = data
        else:
            metavar = _varname_in_stack(data, 1)

    # create a view controller
    if VIEW is None:
        if not detach:
            VIEW = ViewController()
        else:
            VIEW = DetachedViewController()
            VIEW.setDaemon(True)
            VIEW.start()
            if VIEW.is_detached():
                atexit.register(VIEW.exit)
            else:
                VIEW = None
                return None

    # actually show the data
    view_kwargs = {'hdr_rows': hdr_rows, 'idx_cols': idx_cols,
                   'start_pos': start_pos, 'metavar': metavar, 'title': title}
    VIEW.view(model, view_kwargs, wait=wait, recycle=recycle)
    return VIEW
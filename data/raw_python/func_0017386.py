def fedit(data, title="", comment="", icon=None, parent=None, apply=None,
          ok=True, cancel=True, result='list', outfile=None, type='form',
          scrollbar=False, background_color=None, widget_color=None):
    """
    Create form dialog and return result
    (if Cancel button is pressed, return None)

    :param tuple data: datalist, datagroup (see below)
    :param str title: form title
    :param str comment: header comment
    :param QIcon icon: dialog box icon
    :param QWidget parent: parent widget
    :param str ok: customized ok button label
    :param str cancel: customized cancel button label
    :param tuple apply: (label, function) customized button label and callback
    :param function apply: function taking two arguments (result, widgets)
    :param str result: result serialization ('list', 'dict', 'OrderedDict',
                                             'JSON' or 'XML')
    :param str outfile: write result to the file outfile.[py|json|xml]
    :param str type: layout type ('form' or 'questions')
    :param bool scrollbar: vertical scrollbar
    :param str background_color: color of the background
    :param str widget_color: color of the widgets

    :return: Serialized result (data type depends on `result` parameter)
    
    datalist: list/tuple of (field_name, field_value)
    datagroup: list/tuple of (datalist *or* datagroup, title, comment)
    
    Tips:
      * one field for each member of a datalist
      * one tab for each member of a top-level datagroup
      * one page (of a multipage widget, each page can be selected with a 
        combo box) for each member of a datagroup inside a datagroup
       
    Supported types for field_value:
      - int, float, str, unicode, bool
      - colors: in Qt-compatible text form, i.e. in hex format or name (red,...)
                (automatically detected from a string)
      - list/tuple:
          * the first element will be the selected index (or value)
          * the other elements can be couples (key, value) or only values
    """
    # Create a QApplication instance if no instance currently exists
    # (e.g. if the module is used directly from the interpreter)
    test_travis = os.environ.get('TEST_CI_WIDGETS', None)
    if test_travis is not None:
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        timer = QTimer(app)
        timer.timeout.connect(app.quit)
        timer.start(1000)
    elif QApplication.startingUp():
        _app = QApplication([])
        translator_qt = QTranslator()
        translator_qt.load('qt_' + QLocale.system().name(),
                       QLibraryInfo.location(QLibraryInfo.TranslationsPath))
        _app.installTranslator(translator_qt)

    serial = ['list', 'dict', 'OrderedDict', 'JSON', 'XML']
    if result not in serial:
        print("Warning: '%s' not in %s, default to list" %
              (result, ', '.join(serial)), file=sys.stderr)
        result = 'list'

    layouts = ['form', 'questions']
    if type not in layouts:
        print("Warning: '%s' not in %s, default to form" %
              (type, ', '.join(layouts)), file=sys.stderr)
        type = 'form'

    dialog = FormDialog(data, title, comment, icon, parent, apply, ok, cancel,
                        result, outfile, type, scrollbar, background_color,
                        widget_color)
    if dialog.exec_():
        return dialog.get()
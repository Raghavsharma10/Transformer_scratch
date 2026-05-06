def handle_extensions(extensions=None, ignored=None):
    """
    Organizes multiple extensions that are separated with commas or passed by
    using --extension/-e multiple times. Note that the .py extension is ignored
    here because of the way non-*.py files are handled in ``extract`` messages
    (they are copied to file.ext.py files to trick xgettext to parse them as
     Python files).

    For example: running::

        $ verboselib-manage extract -e js,txt -e xhtml -a

    would result in an extension list ``['.js', '.txt', '.xhtml']``

    .. code-block:: python

        >>> handle_extensions(['.html', 'html,js,py,py,py,.py', 'py,.py'])
        set(['.html', '.js'])
        >>> handle_extensions(['.html, txt,.tpl'])
        set(['.html', '.tpl', '.txt'])

    Taken `from Django <http://bit.ly/1r7Eokw>`_ and changed a bit.
    """
    extensions = extensions or ()
    ignored = ignored or ('py', )

    ext_list = []
    for ext in extensions:
        ext_list.extend(ext.replace(' ', '').split(','))
    for i, ext in enumerate(ext_list):
        if not ext.startswith('.'):
            ext_list[i] = '.%s' % ext_list[i]
    return set([x for x in ext_list if x.strip('.') not in ignored])
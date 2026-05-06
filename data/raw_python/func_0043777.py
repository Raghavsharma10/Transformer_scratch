def view_include(view_module, namespace=None, app_name=None):
    """
    Includes view in the url, works similar to django include function.
    Auto imports all class based views that are subclass of ``URLView`` and
    all functional views that have been decorated with ``url_view``.

    :param view_module: object of the module or string with importable path
    :param namespace: name of the namespaces, it will be guessed otherwise
    :param app_name: application name
    :return: result of urls.include
    """

    # since Django 1.8 patterns() are deprecated, list should be used instead
    # {priority:[views,]}
    view_dict = defaultdict(list)

    if isinstance(view_module, six.string_types):
        view_module = importlib.import_module(view_module)

    # pylint:disable=unused-variable
    for member_name, member in inspect.getmembers(view_module):
        is_class_view = inspect.isclass(member) and issubclass(member, URLView)
        is_func_view = (inspect.isfunction(member) and
                        hasattr(member, 'urljects_view') and
                        member.urljects_view)

        if (is_class_view and member is not URLView) or is_func_view:
            view_dict[member.url_priority].append(
                url(member.url, member, name=member.url_name))

    view_patterns = list(*[
        view_dict[priority] for priority in sorted(view_dict)
        ])

    return urls.include(
        arg=view_patterns,
        namespace=namespace,
        app_name=app_name)
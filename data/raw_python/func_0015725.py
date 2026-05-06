def get_type_name(type_):
    """Gives a name for a type that is suitable for a docstring.

    int -> "int"
    Gtk.Window -> "Gtk.Window"
    [int] -> "[int]"
    {int: Gtk.Button} -> "{int: Gtk.Button}"
    """

    if type_ is None:
        return ""
    if isinstance(type_, string_types):
        return type_
    elif isinstance(type_, list):
        assert len(type_) == 1
        return "[%s]" % get_type_name(type_[0])
    elif isinstance(type_, dict):
        assert len(type_) == 1
        key, value = list(type_.items())[0]
        return "{%s: %s}" % (get_type_name(key), get_type_name(value))
    elif type_.__module__ in ("__builtin__", "builtins"):
        return type_.__name__
    else:
        return "%s.%s" % (type_.__module__, type_.__name__)
def replace_fields(text, context, autoescape=None, errors='inline'):
    """
    Allow simple field replacements, using the python str.format() syntax.

    When a string is passed that is tagged with :func:`~django.utils.safestring.mark_safe`,
    the context variables will be escaped before replacement.

    This function is used instead of lazily using Django templates,
    which can also the {% load %} stuff and {% include %} things.
    """
    raise_errors = errors == 'raise'
    ignore_errors = errors == 'ignore'
    inline_errors = errors == 'inline'

    if autoescape is None:
        # When passing a real template context, use it's autoescape setting.
        # Otherwise, default to true.
        autoescape = getattr(context, 'autoescape', True)

    is_safe_string = isinstance(text, SafeData)
    if is_safe_string and autoescape:
        escape_function = conditional_escape
        escape_error = lambda x: u"<span style='color:red;'>{0}</span>".format(x)
    else:
        escape_function = force_text
        escape_error = six.text_type

    # Using str.format() may raise a KeyError when some fields are not provided.
    # Instead, simulate its' behavior to make sure all items that were found will be replaced.
    start = 0
    new_text = []
    for match in RE_FORMAT.finditer(text):
        new_text.append(text[start:match.start()])
        start = match.end()

        # See if the element was found
        key = match.group('var')
        try:
            value = context[key]
        except KeyError:
            logger.debug("Missing key %s in email template %s!", key, match.group(0))
            if raise_errors:
                raise
            elif ignore_errors:
                new_text.append(match.group(0))
            elif inline_errors:
                new_text.append(escape_error("!!missing {0}!!".format(key)))
            continue

        # See if further processing is needed.
        attr = match.group('attr')
        if attr:
            try:
                value = getattr(value, attr)
            except AttributeError:
                logger.debug("Missing attribute %s in email template %s!", attr, match.group(0))
                if raise_errors:
                    raise
                elif ignore_errors:
                    new_text.append(match.group(0))
                elif inline_errors:
                    new_text.append(escape_error("!!invalid attribute {0}.{1}!!".format(key, attr)))
                continue

        format = match.group('format')
        if format:
            try:
                template = u"{0" + format + "}"
                value = template.format(value)
            except ValueError:
                logger.debug("Invalid format %s in email template %s!", format, match.group(0))
                if raise_errors:
                    raise
                elif ignore_errors:
                    new_text.append(match.group(0))
                elif inline_errors:
                    new_text.append(escape_error("!!invalid format {0}!!".format(format)))
                continue
        else:
            value = escape_function(value)

        # Add the value
        new_text.append(value)

    # Add remainder, and join
    new_text.append(text[start:])
    new_text = u"".join(new_text)

    # Convert back to safestring if it was passed that way
    if is_safe_string:
        return mark_safe(new_text)
    else:
        return new_text
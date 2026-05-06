def autoescape(context, nodelist, setting):
    """
    Force autoescape behaviour for this block.
    """
    old_setting = context.autoescape
    context.autoescape = setting
    output = nodelist.render(context)
    context.autoescape = old_setting
    if setting:
        return mark_safe(output)
    else:
        return output
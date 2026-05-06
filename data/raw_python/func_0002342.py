def apply_pre_filters(instance, html):
    """
    Perform optimizations in the HTML source code.

    :type instance: fluent_contents.models.ContentItem
    :raise ValidationError: when one of the filters detects a problem.
    """
    # Allow pre processing. Typical use-case is HTML syntax correction.
    for post_func in appsettings.PRE_FILTER_FUNCTIONS:
        html = post_func(instance, html)

    return html
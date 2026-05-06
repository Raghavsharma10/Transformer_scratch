def apply_post_filters(instance, html):
    """
    Allow post processing functions to change the text.
    This change is not saved in the original text.

    :type instance: fluent_contents.models.ContentItem
    :raise ValidationError: when one of the filters detects a problem.
    """
    for post_func in appsettings.POST_FILTER_FUNCTIONS:
        html = post_func(instance, html)

    return html
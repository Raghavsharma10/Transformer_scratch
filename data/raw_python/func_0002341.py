def apply_filters(instance, html, field_name):
    """
    Run all filters for a given HTML snippet.
    Returns the results of the pre-filter and post-filter as tuple.

    This function can be called from the :meth:`~django.db.models.Model.full_clean` method in the model.
    That function is called when the form values are assigned.
    For example:

    .. code-block:: python

        def full_clean(self, *args, **kwargs):
            super(TextItem, self).full_clean(*args, **kwargs)
            self.html, self.html_final = apply_filters(self, self.html, field_name='html')

    :type instance: fluent_contents.models.ContentItem
    :raise ValidationError: when one of the filters detects a problem.
    """
    try:
        html = apply_pre_filters(instance, html)

        # Perform post processing. This does not effect the original 'html'
        html_final = apply_post_filters(instance, html)
    except ValidationError as e:
        if hasattr(e, 'error_list'):
            # The filters can raise a "dump" ValidationError with a single error.
            # However, during post_clean it's expected that the fields are named.
            raise ValidationError({
                field_name: e.error_list
            })
        raise

    return html, html_final
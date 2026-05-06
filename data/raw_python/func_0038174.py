def update_contributions(sender, instance, action, model, pk_set, **kwargs):
    """Creates a contribution for each author added to an article.
    """
    if action != 'pre_add':
        return
    else:
        for author in model.objects.filter(pk__in=pk_set):
            update_content_contributions(instance, author)
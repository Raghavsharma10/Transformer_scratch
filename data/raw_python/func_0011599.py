def choice_voters_changed_update_cache(
        sender, instance, action, reverse, model, pk_set, **kwargs):
    """Update cache when choice.voters changes."""
    if action not in ('post_add', 'post_remove', 'post_clear'):
        # post_clear is not handled, because clear is called in
        # django.db.models.fields.related.ReverseManyRelatedObjects.__set__
        # before setting the new order
        return

    if model == User:
        assert type(instance) == Choice
        choices = [instance]
        if pk_set:
            users = list(User.objects.filter(pk__in=pk_set))
        else:
            users = []
    else:
        if pk_set:
            choices = list(Choice.objects.filter(pk__in=pk_set))
        else:
            choices = []
        users = [instance]

    from .tasks import update_cache_for_instance
    for choice in choices:
        update_cache_for_instance('Choice', choice.pk, choice)
    for user in users:
        update_cache_for_instance('User', user.pk, user)
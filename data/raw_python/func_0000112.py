def process_actions(action_ids=None):
    """
    Process actions in the publishing schedule.

    Returns the number of actions processed.
    """
    actions_taken = 0
    action_list = PublishAction.objects.prefetch_related(
        'content_object',
    ).filter(
        scheduled_time__lte=timezone.now(),
    )

    if action_ids is not None:
        action_list = action_list.filter(id__in=action_ids)

    for action in action_list:
        action.process_action()
        action.delete()
        actions_taken += 1

    return actions_taken
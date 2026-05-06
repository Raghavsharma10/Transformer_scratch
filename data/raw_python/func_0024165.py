def send_message_for_lane_change(sender, **kwargs):
    """
    Sends a message to possible owners of the current workflows
     next lane.

    Args:
        **kwargs: ``current`` and ``possible_owners`` are required.
        sender (User): User object
    """
    current = kwargs['current']
    owners = kwargs['possible_owners']
    if 'lane_change_invite' in current.task_data:
        msg_context = current.task_data.pop('lane_change_invite')
    else:
        msg_context = DEFAULT_LANE_CHANGE_INVITE_MSG

    wfi = WFCache(current).get_instance()

    # Deletion of used passive task invitation which belongs to previous lane.
    TaskInvitation.objects.filter(instance=wfi, role=current.role, wf_name=wfi.wf.name).delete()

    today = datetime.today()
    for recipient in owners:
        inv = TaskInvitation(
            instance=wfi,
            role=recipient,
            wf_name=wfi.wf.name,
            progress=30,
            start_date=today,
            finish_date=today + timedelta(15)
        )
        inv.title = current.task_data.get('INVITATION_TITLE') or wfi.wf.title
        inv.save()

        # try to send notification, if it fails go on
        try:

            recipient.send_notification(title=msg_context['title'],
                                        message="%s %s" % (wfi.wf.title, msg_context['body']),
                                        typ=1,  # info
                                        url='',
                                        sender=sender
                                        )
        except: # todo: specify which exception
            pass
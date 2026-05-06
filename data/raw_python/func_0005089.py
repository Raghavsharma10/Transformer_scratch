def admin_log(instances, msg: str, who: User=None, **kw):
    """
    Logs an entry to admin logs of model(s).
    :param instances: Model instance or list of instances
    :param msg: Message to log
    :param who: Who did the change
    :param kw: Optional key-value attributes to append to message
    :return: None
    """

    from django.contrib.admin.models import LogEntry, CHANGE
    from django.contrib.admin.options import get_content_type_for_model
    from django.utils.encoding import force_text

    # use system user if 'who' is missing
    if not who:
        username = settings.DJANGO_SYSTEM_USER if hasattr(settings, 'DJANGO_SYSTEM_USER') else 'system'
        who, created = User.objects.get_or_create(username=username)

    # append extra keyword attributes if any
    att_str = ''
    for k, v in kw.items():
        if hasattr(v, 'pk'):  # log only primary key for model instances, not whole str representation
            v = v.pk
        att_str += '{}={}'.format(k, v) if not att_str else ', {}={}'.format(k, v)
    if att_str:
        att_str = ' [{}]'.format(att_str)
    msg = str(msg) + att_str

    if not isinstance(instances, list) and not isinstance(instances, tuple):
        instances = [instances]
    for instance in instances:
        if instance:
            LogEntry.objects.log_action(
                user_id=who.pk,
                content_type_id=get_content_type_for_model(instance).pk,
                object_id=instance.pk,
                object_repr=force_text(instance),
                action_flag=CHANGE,
                change_message=msg,
            )
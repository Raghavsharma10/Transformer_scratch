def purge_old_event_logs(delete_before_days=7):
    """
    Purges old event logs from the database table
    """
    delete_before_date = timezone.now() - timedelta(days=delete_before_days)
    logs_deleted = EventLog.objects.filter(
        created_on__lte=delete_before_date).delete()
    return logs_deleted
def purge_old_logs(delete_before_days=7):
    """
    Purges old logs from the database table
    """
    delete_before_date = timezone.now() - timedelta(days=delete_before_days)
    logs_deleted = Log.objects.filter(
        created_on__lte=delete_before_date).delete()
    return logs_deleted
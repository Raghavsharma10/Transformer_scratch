def purge_old_request_logs(delete_before_days=7):
    """
    Purges old request logs from the database table
    """
    delete_before_date = timezone.now() - timedelta(days=delete_before_days)
    logs_deleted = RequestLog.objects.filter(
        created_on__lte=delete_before_date).delete()
    return logs_deleted
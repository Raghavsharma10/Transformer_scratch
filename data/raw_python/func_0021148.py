def most_seen_creators_by_works(work_kind=None, role_name=None, num=10):
    """
    Returns a QuerySet of the Creators that are associated with the most Works.
    """
    return Creator.objects.by_works(kind=work_kind, role_name=role_name)[:num]
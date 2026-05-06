def get_revisions(page, page_num=1):
    """
    Returns paginated queryset of PageRevision instances for
    specified Page instance.

    :param page: the page instance.
    :param page_num: the pagination page number.
    :rtype: django.db.models.query.QuerySet.
    """
    revisions   = page.revisions.order_by('-created_at')
    current     = page.get_latest_revision()

    if current:
        revisions.exclude(id=current.id)

    paginator = Paginator(revisions, 5)

    try:
        revisions = paginator.page(page_num)
    except PageNotAnInteger:
        revisions = paginator.page(1)
    except EmptyPage:
        revisions = paginator.page(paginator.num_pages)

    return revisions
def _paginate(self, current_page, query_set, per_page=10):
    """
    Handles pagination of object listings.

    Args:
        current_page int:
            Current page number
        query_set (:class:`QuerySet<pyoko:pyoko.db.queryset.QuerySet>`):
            Object listing queryset.
        per_page int:
            Objects per page.

    Returns:
        QuerySet object, pagination data dict as a tuple
    """
    total_objects = query_set.count()
    total_pages = int(total_objects / per_page or 1)
    # add orphans to last page
    current_per_page = per_page + (
        total_objects % per_page if current_page == total_pages else 0)
    pagination_data = dict(page=current_page,
                           total_pages=total_pages,
                           total_objects=total_objects,
                           per_page=current_per_page)
    query_set = query_set.set_params(rows=current_per_page, start=(current_page - 1) * per_page)
    return query_set, pagination_data
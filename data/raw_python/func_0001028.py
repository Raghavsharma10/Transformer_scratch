def compat_pagination_messages(cls):
    """
    For DRF 3.1 and higher, pagination is defined at the paginator level (see http://www.django-rest-framework.org/topics/3.2-announcement/).
    For DRF 3.0 and lower, it can be handled at the view level.
    """
    if DRFVLIST[0] == 3 and DRFVLIST[1] >= 1:
        setattr(cls, "pagination_class", MessagePagination)
        return cls
    else:
        # DRF 2 pagination
        setattr(cls, "paginate_by", getattr(settings, "DJANGO_REST_MESSAGING_MESSAGES_PAGE_SIZE", 30))
        return cls
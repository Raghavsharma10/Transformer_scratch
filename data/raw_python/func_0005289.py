def _meta_get_resource_sync(md_uuid):
    """Just a meta func to get execution time"""
    isogeo.resource(id_resource=md_uuid)

    elapsed = default_timer() - START_TIME
    time_completed_at = "{:5.2f}s".format(elapsed)
    print("{0:<30} {1:>20}".format(md_uuid, time_completed_at))

    return
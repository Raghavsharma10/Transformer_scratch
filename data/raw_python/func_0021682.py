def post_save_stop(sender, instance, **kwargs):
    '''Update related objects when the Stop is updated'''
    from multigtfs.models.trip import Trip
    trip_ids = instance.stoptime_set.filter(
        trip__shape=None).values_list('trip_id', flat=True).distinct()
    for trip in Trip.objects.filter(id__in=trip_ids):
        trip.update_geometry()
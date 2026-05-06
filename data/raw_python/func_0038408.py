def update_pzone(**kwargs):
    """Update pzone data in the DB"""

    pzone = PZone.objects.get(**kwargs)

    # get the data and loop through operate_on, applying them if necessary
    when = timezone.now()
    data = pzone.data
    for operation in pzone.operations.filter(when__lte=when, applied=False):
        data = operation.apply(data)
        operation.applied = True
        operation.save()
    pzone.data = data

    # create a history entry
    pzone.history.create(data=pzone.data)

    # save modified pzone, making transactions permanent
    pzone.save()
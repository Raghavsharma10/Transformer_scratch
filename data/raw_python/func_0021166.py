def forwards(apps, schema_editor):
    """
    Having added the new 'exhibition' Work type, we're going to assume that
    every Event of type 'museum' should actually have one Exhibition attached.

    So, we'll add one, with the same title as the Event.
    And we'll move all Creators from the Event to the Exhibition.
    """
    Event = apps.get_model('spectator_events', 'Event')
    Work = apps.get_model('spectator_events', 'Work')
    WorkRole = apps.get_model('spectator_events', 'WorkRole')
    WorkSelection = apps.get_model('spectator_events', 'WorkSelection')

    for event in Event.objects.filter(kind='museum'):

        # Create a new Work based on this Event's details.

        work = Work.objects.create(
            kind='exhibition',
            title=event.title,
            title_sort=event.title_sort
        )
        # This doesn't generate the slug field automatically because Django.
        # So we'll have to do it manually. Graarhhh.
        work.slug = generate_slug(work.pk)
        work.save()

        # Associate the new Work with the Event.
        WorkSelection.objects.create(
            event=event,
            work=work
        )

        # Associate any Creators on the Event with the new Work.
        for role in event.roles.all():
            WorkRole.objects.create(
                creator=role.creator,
                work=work,
                role_name=role.role_name,
                role_order=role.role_order
            )

            # Remove Creators from the Event.
            role.delete()
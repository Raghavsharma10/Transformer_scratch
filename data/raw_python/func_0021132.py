def forward(apps, schema_editor):
    """
    Copying data from the old `Event.movie` and `Event.play` ForeignKey fields
    into the new `Event.movies` and `Event.plays` ManyToManyFields.
    """

    Event = apps.get_model('spectator_events', 'Event')
    MovieSelection = apps.get_model('spectator_events', 'MovieSelection')
    PlaySelection = apps.get_model('spectator_events', 'PlaySelection')

    for event in Event.objects.all():
        if event.movie is not None:
            selection = MovieSelection(event=event, movie=event.movie)
            selection.save()

        if event.play is not None:
            selection = PlaySelection(event=event, play=event.play)
            selection.save()
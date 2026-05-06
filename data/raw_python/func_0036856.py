def retarget_to_length(song, duration, start=True, end=True, slack=5,
                       beats_per_measure=None):
    """Create a composition of a song that changes its length
    to a given duration.

    :param song: Song to retarget
    :type song: :py:class:`radiotool.composer.Song`
    :param duration: Duration of retargeted song (in seconds)
    :type duration: float
    :param start: Start the retargeted song at the
                  beginning of the original song
    :type start: boolean
    :param end: End the retargeted song at the end of the original song
    :type end: boolean
    :param slack: Track will be within slack seconds of the target
                  duration (more slack allows for better-sounding music)
    :type slack: float
    :returns: Composition of retargeted song
    :rtype: :py:class:`radiotool.composer.Composition`
    """

    duration = float(duration)

    constraints = [
        rt_constraints.TimbrePitchConstraint(
            context=0, timbre_weight=1.0, chroma_weight=1.0),
        rt_constraints.EnergyConstraint(penalty=.5),
        rt_constraints.MinimumLoopConstraint(8),
    ]

    if beats_per_measure is not None:
        constraints.append(
            rt_constraints.RhythmConstraint(beats_per_measure, .125))

    if start:
        constraints.append(
            rt_constraints.StartAtStartConstraint(padding=0))

    if end:
        constraints.append(
            rt_constraints.EndAtEndConstraint(padding=slack))

    comp, info = retarget(
        [song], duration, constraints=[constraints],
        fade_in_len=None, fade_out_len=None)

    # force the new track to extend to the end of the song
    if end:
        last_seg = sorted(
            comp.segments,
            key=lambda seg:
            seg.comp_location_in_seconds + seg.duration_in_seconds
        )[-1]

        last_seg.duration_in_seconds = (
            song.duration_in_seconds - last_seg.start_in_seconds)

    path_cost = info["path_cost"]
    total_nonzero_cost = []
    total_nonzero_points = []
    for node in path_cost:
        if float(node.name) > 0.0:
            total_nonzero_cost.append(float(node.name))
            total_nonzero_points.append(float(node.time))

    transitions = zip(total_nonzero_points, total_nonzero_cost)

    for transition in transitions:
        comp.add_label(Label("crossfade", transition[0]))
    return comp
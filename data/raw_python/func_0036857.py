def retarget_with_change_points(song, cp_times, duration):
    """Create a composition of a song of a given duration that reaches
    music change points at specified times. This is still under
    construction. It might not work as well with more than
    2 ``cp_times`` at the moment.

    Here's an example of retargeting music to be 40 seconds long and
    hit a change point at the 10 and 30 second marks::

        song = Song("instrumental_music.wav")
        composition, change_points =\
            retarget.retarget_with_change_points(song, [10, 30], 40)
        composition.export(filename="retargeted_instrumental_music.")

    :param song: Song to retarget
    :type song: :py:class:`radiotool.composer.Song`
    :param cp_times: Times to reach change points (in seconds)
    :type cp_times: list of floats
    :param duration: Target length of retargeted music (in seconds)
    :type duration: float
    :returns: Composition of retargeted song and list of locations of
        change points in the retargeted composition
    :rtype: (:py:class:`radiotool.composer.Composition`, list)
    """
    analysis = song.analysis

    beat_length = analysis[BEAT_DUR_KEY]
    beats = np.array(analysis["beats"])

    # find change points
    cps = np.array(novelty(song, nchangepoints=4))
    cp_times = np.array(cp_times)

    # mark change points in original music
    def music_labels(t):
        # find beat closest to t
        closest_beat_idx = np.argmin(np.abs(beats - t))
        closest_beat = beats[closest_beat_idx]
        closest_cp = cps[np.argmin(np.abs(cps - closest_beat))]

        if np.argmin(np.abs(beats - closest_cp)) == closest_beat_idx:
            return "cp"
        else:
            return "noncp"

    # mark where we want change points in the output music
    # (a few beats of slack to improve the quality of the end result)
    def out_labels(t):
        if np.min(np.abs(cp_times - t)) < 1.5 * beat_length:
            return "cp"
        return "noncp"

    m_labels = [music_labels(i) for i in
                np.arange(0, song.duration_in_seconds, beat_length)]
    o_labels = [out_labels(i) for i in np.arange(0, duration, beat_length)]

    constraints = [
        rt_constraints.TimbrePitchConstraint(
            context=0, timbre_weight=1.0, chroma_weight=1.0),
        rt_constraints.EnergyConstraint(penalty=.5),
        rt_constraints.MinimumLoopConstraint(8),
        rt_constraints.NoveltyConstraint(m_labels, o_labels, 1.0)
    ]

    comp, info = retarget(
        [song], duration, constraints=[constraints],
        fade_in_len=None, fade_out_len=None)

    final_cp_locations = [beat_length * i
                          for i, label in enumerate(info['result_labels'])
                          if label == 'cp']

    return comp, final_cp_locations
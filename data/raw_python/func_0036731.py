def add_music_cue(self, track, comp_cue, song_cue, duration=6.0,
                      padding_before=12.0, padding_after=12.0):
        """Add a music cue to the composition. This doesn't do any audio
        analysis, it just aligns a specified point in the track
        (presumably music) with a location in the composition. See
        UnderScore_ for a visualization of what this is doing to the
        music track.

        .. _UnderScore: http://vis.berkeley.edu/papers/underscore/

        :param track: Track to align in the composition
        :type track: :py:class:`radiotool.composer.Track`
        :param float comp_cue: Location in composition to align music cue (in seconds)
        :param float song_cue: Location in the music track to align with the composition cue (in seconds)
        :param float duration: Duration of music after the song cue before the music starts to fade out (in seconds)
        :param float padding_before: Duration of music playing softly before the music cue/composition cue (in seconds)
        :param float padding_after: Duration of music playing softly after the music cue/composition cue (in seconds)
        """

        self.tracks.add(track)
        
        pre_fade = 3
        post_fade = 3
        
        if padding_before + pre_fade > song_cue:
            padding_before = song_cue - pre_fade
            
        if padding_before + pre_fade > score_cue:
            padding_before = score_cue - pre_fade

        s = Segment(track, score_cue - padding_before - pre_fade,
                    song_cue - padding_before - pre_fade,
                    pre_fade + padding_before + duration +
                    padding_after + post_fade)

        self.add_segment(s)
        
        d = []

        dyn_adj = 1
        
        track.current_frame = 0
        
        d.append(Fade(track, score_cue - padding_before - pre_fade, pre_fade,
                      0, .1*dyn_adj, fade_type="linear"))

        d.append(Fade(track, score_cue - padding_before, padding_before,
                      .1*dyn_adj, .4*dyn_adj, fade_type="exponential"))

        d.append(Volume(track, score_cue, duration, .4*dyn_adj))

        d.append(Fade(track, score_cue + duration, padding_after,
                      .4*dyn_adj, 0, fade_type="exponential"))

        d.append(Fade(track, score_cue + duration + padding_after, post_fade,
                      .1*dyn_adj, 0, fade_type="linear"))
        self.add_dynamics(d)
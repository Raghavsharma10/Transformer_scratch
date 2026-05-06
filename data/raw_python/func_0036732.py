def build(self, track_list=None, adjust_dynamics=False,
        min_length=None, channels=None):
        """
        Create a numpy array from the composition.

        :param track_list: List of tracks to include in composition generation (``None`` means all tracks will be used)
        :type track_list: list of :py:class:`radiotool.composer.Track`
        :param int min_length: Minimum length of output array (in frames). Will zero pad extra length.
        :param bool. adjust_dynamics: Automatically adjust dynamics. Will document later.
        """
        if track_list is None:
            track_list = self.tracks

        if channels is None:
            channels = self.channels

        parts = {}
        starts = {}
        
        # for universal volume adjustment
        all_frames = np.array([])
        song_frames = np.array([])
        speech_frames = np.array([])
        
        longest_part = max([x.comp_location + x.duration
                            for x in self.segments])
        if len(self.dynamics) > 0:
            longest_part = max((longest_part,
                max([x.comp_location + x.duration
                     for x in self.dynamics])))
        
        for track_idx, track in enumerate(track_list):
            segments = sorted([v for v in self.segments if v.track == track], 
                              key=lambda k: k.comp_location + k.duration)
            dyns = sorted([d for d in self.dynamics if d.track == track],
                           key=lambda k: k.comp_location)

            if len(segments) > 0:
                start_loc = min([x.comp_location for x in segments])
                end_loc = max([x.comp_location + x.duration for x in segments])
                if len(dyns) > 0:
                    start_loc = min((start_loc, 
                         min([d.comp_location for d in dyns])))
                    end_loc = max((end_loc,
                        max([d.comp_location + d.duration for d in dyns])))
                
                starts[track] = start_loc

                parts[track] = np.zeros((end_loc - start_loc, channels))
                
                for s in segments:

                    frames = s.get_frames(channels=channels).\
                        reshape(-1, channels)
                    
                    # for universal volume adjustment
                    if adjust_dynamics:
                        all_frames = np.append(all_frames,
                            self._remove_end_silence(frames.flatten()))
                        if isinstance(track, Song):
                            song_frames = np.append(song_frames,
                                self._remove_end_silence(frames.flatten()))
                        elif isinstance(track, Speech):
                            speech_frames = np.append(speech_frames,
                                self._remove_end_silence(frames.flatten()))
                                
                    parts[track][s.comp_location - start_loc:
                                 s.comp_location - start_loc + s.duration,
                                 :] = frames

            for d in dyns:
                vol_frames = d.to_array(channels)

                parts[track][d.comp_location - start_loc :
                             d.comp_location - start_loc + d.duration,
                             :] *= vol_frames


        if adjust_dynamics:
            total_energy = RMS_energy(all_frames)
            song_energy = RMS_energy(song_frames)
            speech_energy = RMS_energy(speech_frames)
                
        # dyn_adj = 0.10 / total_energy
        # dyn_adj = speech_energy / sqrt(song_energy) * 5
        if adjust_dynamics:
            if not np.isnan(speech_energy) and not np.isnan(song_energy):
                dyn_adj = sqrt(speech_energy / song_energy) * 1.15
            else:
                dyn_adj = 1
        else:
            dyn_adj = 1

        if longest_part < min_length:
            longest_part = min_length
        out = np.zeros((longest_part, channels))
        for track, part in parts.iteritems():
            out[starts[track]:starts[track] + len(part)] += part

        return out
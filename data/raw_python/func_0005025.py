def inverse(self, encoded, downbeat=None, duration=None):
        '''Inverse transformation for beats and optional downbeats'''

        ann = jams.Annotation(namespace=self.namespace, duration=duration)

        beat_times = np.asarray([t for t, _ in self.decode_events(encoded,
                                                                  transition=self.beat_transition,
                                                                  p_init=self.beat_p_init,
                                                                  p_state=self.beat_p_state) if _])
        beat_frames = time_to_frames(beat_times,
                                     sr=self.sr,
                                     hop_length=self.hop_length)

        if downbeat is not None:
            downbeat_times = set([t for t, _ in self.decode_events(downbeat,
                                                                   transition=self.down_transition,
                                                                   p_init=self.down_p_init,
                                                                   p_state=self.down_p_state) if _])
            pickup_beats = len([t for t in beat_times
                                if t < min(downbeat_times)])
        else:
            downbeat_times = set()
            pickup_beats = 0

        value = - pickup_beats - 1
        for beat_t, beat_f in zip(beat_times, beat_frames):
            if beat_t in downbeat_times:
                value = 1
            else:
                value += 1
            confidence = encoded[beat_f]
            ann.append(time=beat_t,
                       duration=0,
                       value=value,
                       confidence=confidence)

        return ann
def contract(self, time, duration, min_contraction=0.0):
        """Remove empty gaps from the composition starting at a given
        time for a given duration.

        

        """

        # remove audio from the composition starting at time
        # for duration

        contract_dur = 0.0
        contract_start = time

        if self.empty_over_span(time, duration):
            contract_dur = duration
            contract_start = time
        else:
            starts = [s.comp_location_in_seconds for s in self.segments]
            ends = [s.comp_location_in_seconds + s.duration_in_seconds
                    for s in self.segments]

            key_starts = []
            key_ends = []

            for start in starts:
                if start >= time and start < time + duration:
                    # does a segment cover the location right before this start?
                    is_key_start = True
                    for seg in self.segments:
                        if seg.comp_location_in_seconds < start and\
                            seg.comp_location_in_seconds + seg.duration_in_seconds >= start:
                            is_key_start = False
                            break
                    if is_key_start:
                        key_starts.append(start)

            for end in ends:
                if end >= time and end < time + duration:
                    # does a segment cover the location right before this start?
                    is_key_end = True
                    for seg in self.segments:
                        if seg.comp_location_in_seconds <= end and\
                            seg.comp_location_in_seconds + seg.duration_in_seconds > end:
                            is_key_end = False
                            break
                    if is_key_end:
                        key_ends.append(end)

            if len(key_starts) + len(key_ends) == 0: return 0, 0

            # combine key starts and key ends
            key_both = [s for s in key_starts]
            key_both.extend([s for s in key_ends])
            key_both = sorted(key_both)

            first_key = key_both[0]
            if first_key in key_starts:
                contract_start = time
                contract_dur = first_key - time
            else:
                contract_start = first_key
                if len(key_both) >= 2:
                    contract_dur = key_both[1] - first_key
                else:
                    contract_dur = time + duration - first_key

        if contract_dur > min_contraction:
            for seg in self.segments:
                if seg.comp_location_in_seconds > contract_start:
                    dur_samples = int(seg.samplerate * contract_dur)
                    seg.comp_location -= dur_samples
            for dyn in self.dynamics:
                if dyn.comp_location_in_seconds > contract_start:
                    dur_samples = int(seg.samplerate * contract_dur)
                    dyn.comp_location -= dur_samples
            return contract_start, contract_dur
        else:
            return 0.0, 0.0
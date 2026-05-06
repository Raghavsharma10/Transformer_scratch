def get_start_stops(transcript_sequence, start_codons=None, stop_codons=None):
        """Return start and stop positions for all frames in the given
        transcript.

        """
        transcript_sequence = transcript_sequence.upper()  # for comparison with codons below
        if not start_codons:
            start_codons = ['ATG']
        if not stop_codons:
            stop_codons = ['TAA', 'TAG', 'TGA']

        seq_frames = {1: {'starts': [], 'stops': []},
                      2: {'starts': [], 'stops': []},
                      3: {'starts': [], 'stops': []}}

        for codons, positions in ((start_codons, 'starts'),
                                  (stop_codons, 'stops')):
            if len(codons) > 1:
                pat = re.compile('|'.join(codons))
            else:
                pat = re.compile(codons[0])

            for m in re.finditer(pat, transcript_sequence):
                # Increment position by 1, Frame 1 starts at position 1 not 0
                start = m.start() + 1
                rem = start % 3
                if rem == 1:  # frame 1
                    seq_frames[1][positions].append(start)
                elif rem == 2:  # frame 2
                    seq_frames[2][positions].append(start)
                elif rem == 0:  # frame 3
                    seq_frames[3][positions].append(start)
        return seq_frames
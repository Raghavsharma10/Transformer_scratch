def parse_frames(filename):
    ''' quick and dirty eprime txt file parsing - doesn\'t account for nesting 
    
    **Example usage**::
    
        for frame in neural.eprime.parse_frames("experiment-1.txt"):
            trial_type = frame['TrialSlide.Tag']
            trial_rt = float(frame['TrialSlide.RT'])
            print '%s: %fms' % (trial_type,trial_rt)
    '''
    frames = []
    frame = {}
    data = nl.universal_read(filename)
    lines = [x.strip() for x in data.split('\n')]
    for line in lines:
        if line == '*** LogFrame Start ***':
            frame = {}
        if line == '*** LogFrame End ***':
            frames.append(frame)
            yield frame
        fields = line.split(": ")
        if len(fields)==2:
            frame[fields[0]] = fields[1]
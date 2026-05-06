def join(cls, splits):
        """
        Join an array of ids into a compound id string
        """
        segments = []
        for split in splits:
            segments.append('"{}",'.format(split))
        if len(segments) > 0:
            segments[-1] = segments[-1][:-1]
        jsonString = '[{}]'.format(''.join(segments))
        return jsonString
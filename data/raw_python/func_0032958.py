def swap_twitter_subject(subject, body):
    """If subject starts from 'Tweet from...'
    then we need to get first meaning line from the body."""

    if subject.startswith('Tweet from'):
        lines = body.split('\n')
        for idx, line in enumerate(lines):
            if re.match(r'.*, ?\d{2}:\d{2}]]', line) is not None:
                try:
                    subject = lines[idx + 1]
                except IndexError:
                    pass
                break
    return subject, body
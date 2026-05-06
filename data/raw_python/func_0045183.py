def output_to_file(filename, tracklisting, action):
    """
    Produce requested output; either output text file, tag audio file or do
    both.

    filename: a string of path + filename without file extension
    tracklisting: a string containing a tracklisting
    action: 'tag', 'text' or 'both', from command line arguments
    """
    if action in ('tag', 'both'):
        audio_tagged = tag_audio(filename, tracklisting)
        if action == 'both' and audio_tagged:
            write_text(filename, tracklisting)
    elif action == 'text':
        write_text(filename, tracklisting)
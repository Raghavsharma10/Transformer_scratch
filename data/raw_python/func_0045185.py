def tag_audio(filename, tracklisting):
    """Return True if audio tagged successfully; handle tagging audio."""
    # TODO: maybe actually glob for files, then try tagging if present?
    if not(tag_audio_file(filename + '.m4a', tracklisting) or
           tag_audio_file(filename + '.mp3', tracklisting)):
        print("Cannot find or access any relevant M4A or MP3 audio file.")
        print("Trying to save a text file instead.")
        write_text(filename, tracklisting)
        return False
    return True
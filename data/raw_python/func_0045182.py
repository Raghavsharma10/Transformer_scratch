def tag_audio_file(audio_file, tracklisting):
    """
    Adds tracklisting as list to lyrics tag of audio file if not present.
    Returns True if successful or not needed, False if tagging fails.
    """
    try:
        save_tag_to_audio_file(audio_file, tracklisting)
    # TODO: is IOError required now or would the mediafile exception cover it?
    except (IOError, mediafile.UnreadableFileError):
        print("Unable to save tag to file:", audio_file)
        audio_tagging_successful = False
    except TagNotNeededError:
        audio_tagging_successful = True
    else:
        audio_tagging_successful = True
    return audio_tagging_successful
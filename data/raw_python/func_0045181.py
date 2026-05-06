def save_tag_to_audio_file(audio_file, tracklisting):
    """
    Saves tag to audio file.
    """
    print("Trying to tag {}".format(audio_file))
    f = mediafile.MediaFile(audio_file)

    if not f.lyrics:
        print("No tracklisting present. Creating lyrics tag.")
        f.lyrics = 'Tracklisting' + '\n' + tracklisting
    elif tracklisting not in f.lyrics:
        print("Appending tracklisting to existing lyrics tag.")
        f.lyrics = f.lyrics + '\n\n' + 'Tracklisting' + '\n' + tracklisting
    else:
        print("Tracklisting already present. Not modifying file.")
        raise TagNotNeededError

    f.save()
    print("Saved tag to file:", audio_file)
def text_to_title(value):
    """when a title is required, generate one from the value"""
    title = None
    if not value:
        return title
    words = value.split(" ")
    keep_words = []
    for word in words:
        if word.endswith(".") or word.endswith(":"):
            keep_words.append(word)
            if len(word) > 1 and "<italic>" not in word and "<i>" not in word:
                break
        else:
            keep_words.append(word)
    if len(keep_words) > 0:
        title = " ".join(keep_words)
        if title.split(" ")[-1] != "spp.":
            title = title.rstrip(" .:")
    return title
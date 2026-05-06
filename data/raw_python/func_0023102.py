def focused_on_tweet(request):
    """
    Return index if focused on tweet False if couldn't
    """
    slots = request.get_slot_map()
    if "Index" in slots and slots["Index"]:
        index = int(slots['Index'])

    elif "Ordinal" in slots and slots["Index"]:
        parse_ordinal = lambda inp : int("".join([l for l in inp if l in string.digits]))
        index = parse_ordinal(slots['Ordinal'])
    else:
        return False
        
    index = index - 1 # Going from regular notation to CS notation
    user_state = twitter_cache.get_user_state(request.access_token())
    queue = user_state['user_queue'].queue()
    if index < len(queue):
        # Analyze tweet in queue
        tweet_to_analyze = queue[index]
        user_state['focus_tweet'] = tweet_to_analyze
        return index + 1 # Returning to regular notation
        twitter_cache.serialize()
    return False
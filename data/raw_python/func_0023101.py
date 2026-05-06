def tweet_list_handler(request, tweet_list_builder, msg_prefix=""):

    """ This is a generic function to handle any intent that reads out a list of tweets"""
    # tweet_list_builder is a function that takes a unique identifier and returns a list of things to say
    tweets = tweet_list_builder(request.access_token())
    print (len(tweets), 'tweets found')
    if tweets:
        twitter_cache.initialize_user_queue(user_id=request.access_token(),
                                            queue=tweets)
        text_to_read_out = twitter_cache.user_queue(request.access_token()).read_out_next(MAX_RESPONSE_TWEETS)        
        message = msg_prefix + text_to_read_out + ", say 'next' to hear more, or reply to a tweet by number."
        return alexa.create_response(message=message,
                                     end_session=False)
    else:
        return alexa.create_response(message="Sorry, no tweets found, please try something else", 
                                 end_session=False)
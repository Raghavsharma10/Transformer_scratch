def next_intent_handler(request):
    """
    Takes care of things whenver the user says 'next'
    """

    message = "Sorry, couldn't find anything in your next queue"
    end_session = True
    if True:
        user_queue = twitter_cache.user_queue(request.access_token())
        if not user_queue.is_finished():
            message = user_queue.read_out_next(MAX_RESPONSE_TWEETS)
            if not user_queue.is_finished():
                end_session = False
                message = message + ". Please, say 'next' if you want me to read out more. "
    return alexa.create_response(message=message,
                                 end_session=end_session)
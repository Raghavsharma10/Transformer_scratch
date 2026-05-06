def post_tweet_intent_handler(request):
    """
    Use the 'intent' field in the VoiceHandler to map to the respective intent.
    """
    tweet = request.get_slot_value("Tweet")
    tweet = tweet if tweet else ""    
    if tweet:
        user_state = twitter_cache.get_user_state(request.access_token())
        def action():
            return post_tweet(request.access_token(), tweet)

        message = "I am ready to post the tweet, {} ,\n Please say yes to confirm or stop to cancel .".format(tweet)
        user_state['pending_action'] = {"action" : action,
                                        "description" : message} 
        return r.create_response(message=message, end_session=False)
    else:
        # No tweet could be disambiguated
        message = " ".join(
            [
                "I'm sorry, I couldn't understand what you wanted to tweet .",
                "Please prepend the message with either post or tweet ."
            ]
        )
        return alexa.create_response(message=message, end_session=False)
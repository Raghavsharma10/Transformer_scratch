def launch_request_handler(request):
    """ Annotate functions with @VoiceHandler so that they can be automatically mapped 
    to request types. Use the 'request_type' field to map them to non-intent requests """

    user_id = request.access_token()
    if user_id in twitter_cache.users():

        user_cache = twitter_cache.get_user_state(user_id)        
        user_cache["amzn_id"]= request.user_id()
        base_message = "Welcome to Twitter, {} . How may I help you today ?".format(user_cache["screen_name"])
        print (user_cache)        
        if 'pending_action' in user_cache:
            base_message += " You have one pending action . "
            print ("Found pending action")
            if 'description' in user_cache['pending_action']:
                print ("Found description")
                base_message += user_cache['pending_action']['description']
        return r.create_response(base_message)

    card = r.create_card(title="Please log into twitter", card_type="LinkAccount")
    return r.create_response(message="Welcome to twitter, looks like you haven't logged in!"
                             " Log in via the alexa app.", card_obj=card,
                             end_session=True)
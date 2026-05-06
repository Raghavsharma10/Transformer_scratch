def post_tweet(user_id, message, additional_params={}):
    """
    Helper function to post a tweet 
    """
    url = "https://api.twitter.com/1.1/statuses/update.json"    
    params = { "status" : message }
    params.update(additional_params)
    r = make_twitter_request(url, user_id, params, request_type='POST')
    print (r.text)
    return "Successfully posted a tweet {}".format(message)
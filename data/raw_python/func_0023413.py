def make_twitter_request(url, user_id, params={}, request_type='GET'):
    """ Generically make a request to twitter API using a particular user's authorization """
    if request_type == "GET":
        return requests.get(url, auth=get_twitter_auth(user_id), params=params)
    elif request_type == "POST":
        return requests.post(url, auth=get_twitter_auth(user_id), params=params)
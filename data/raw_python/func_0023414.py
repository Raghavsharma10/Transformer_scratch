def geo_search(user_id, search_location):
    """
    Search for a location - free form
    """
    url = "https://api.twitter.com/1.1/geo/search.json"
    params =  {"query" : search_location }
    response = make_twitter_request(url, user_id, params).json()
    return response
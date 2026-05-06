def read_out_tweets(processed_tweets, speech_convertor=None):
    """
    Input - list of processed 'Tweets'
    output - list of spoken responses
    """
    return ["tweet number {num} by {user}. {text}.".format(num=index+1, user=user, text=text)
               for index, (user, text) in enumerate(processed_tweets)]
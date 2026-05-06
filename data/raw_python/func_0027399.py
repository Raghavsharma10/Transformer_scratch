def make_request_log_message(**args):
    '''
    Creates a string containing all relevant information
    about a request made to the Handle System, for
    logging purposes.

    :handle: The handle that the request is about.
    :url: The url the request is sent to.
    :headers: The headers sent along with the request.
    :verify: Boolean parameter passed to the requests
        module (https verification).
    :resp: The request's response.
    :op: The library operation during which the request
        was sent.
    :payload: Optional. The payload sent with the request.
    :return: A formatted string.

    '''

    mandatory_args = ['op', 'handle', 'url', 'headers', 'verify', 'resp']
    optional_args = ['payload']
    util.check_presence_of_mandatory_args(args, mandatory_args)
    util.add_missing_optional_args_with_value_none(args, optional_args)

    space = '\n   '
    message = ''
    message += '\n'+args['op']+' '+args['handle']
    message += space+'URL:          '+args['url']
    message += space+'HEADERS:      '+str(args['headers'])
    message += space+'VERIFY:       '+str(args['verify'])
    if 'payload' in args.keys():
        message += space+'PAYLOAD:'+space+str(args['payload'])
    message += space+'RESPONSECODE: '+str(args['resp'].status_code)
    message += space+'RESPONSE:'+space+str(args['resp'].content)
    return message
def create_authentication_string(username, password):
    '''
    Creates an authentication string from the username and password.

    :username: Username.
    :password: Password.
    :return: The encoded string.
    '''

    username_utf8 = username.encode('utf-8')
    userpw_utf8 = password.encode('utf-8')
    username_perc = quote(username_utf8)
    userpw_perc = quote(userpw_utf8)

    authinfostring = username_perc + ':' + userpw_perc
    authinfostring_base64 = base64.b64encode(authinfostring.encode('utf-8')).decode('utf-8')
    return authinfostring_base64
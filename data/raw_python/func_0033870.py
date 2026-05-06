def genMeme(template_id, text0, text1):
    '''
    This function returns the url of the meme with the given
    template, upper text, and lower text using the ImgFlip
    meme generation API. Thanks!

    Returns None if it is unable to generate the meme.
    '''

    username = 'blag'
    password = 'blag'

    api_url = 'https://api.imgflip.com/caption_image'

    payload = {
        'template_id': template_id,
        'username': username,
        'password': password,
        'text0': text0,
    }
    # Add bottom text if provided
    if text1 != '':
        payload['text1'] = text1

    try:
        r = requests.get(api_url, params=payload)
    except ConnectionError:
        time.sleep(1)
        r = requests.get(api_url, params=payload)

    # print(parsed_json)
    parsed_json = json.loads(r.text)

    request_status = parsed_json['success']

    if request_status != True:
        error_msg = parsed_json['error_message']
        print(error_msg)
        return None

    else:
        imgURL = parsed_json['data']['url']
        # print(imgURL)
        return imgURL
def processMeme(imgParams):
    '''
    Wrapper function for genMeme() and findMeme()
    imgParams may be a string of the following forms:
    * 'text0 | text1'
    * 'text0'
    * ' | text1'

    Fails gracefully when it can't find or generate a meme
    by returning an appropriate image url with the failure
    message on it.
    '''

    template_id = findMeme(imgParams)

    if template_id is None:
        print("Couldn't find a suitable match for meme :(")
        return meme_not_supported

    # if template_id exists
    imgParams = imgParams.split('|')

    if len(imgParams) == 2 or len(imgParams) == 1:
        text0 = imgParams[0]

        if len(imgParams) == 2:
            text1 = imgParams[1]    # Bottom text text1 exists
        elif len(imgParams) == 1:
            text1 = ''              # No bottom text

        imgURL = genMeme(template_id, text0, text1)

        if imgURL is None:          # Couldn't generate meme
            print("Couldn't generate meme :(")
            return couldnt_create_meme
        else:                       # Success!
            # print(imgURL)
            return imgURL

    elif len(imgParams) > 2:
        print("Too many lines of captions! Cannot create meme.")
        return too_many_lines

    elif len(imgParams) < 1:        # No top text text0 exists
        print("Too few lines of captions! Cannot create meme.")
        return too_few_lines
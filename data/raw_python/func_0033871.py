def findMeme(inptStr):
    '''
    inptStr may be a string of the following forms:
    * 'text0 | text1'
    * 'text0'

    Returns None if it can't find find a meme from the list given above
    '''

    global meme_id_dict

    testStr = inptStr
    testStr.lower()

    template_id = 0

    '''
    meme_id_dict[i] is of form:
    [meme_tagline, meme_name, template_id]
    '''
    for i in range(len(meme_id_dict)):
        test_words = testStr.strip('|.,?!').split(' ')

        meme_words = meme_id_dict[i][0].split(' ')
        common_words = len(list(set(meme_words).intersection(test_words)))

        if (len(meme_words) >= 4 and common_words >= 3) or (len(meme_words) < 4 and common_words >= 1):
            template_id = meme_id_dict[i][2]
            return template_id

    if template_id == 0:
        return None
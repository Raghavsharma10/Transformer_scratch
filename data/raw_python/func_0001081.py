def is_kana(text):
    ''' Check if a text if written in kana only (hiragana & katakana)
    if text is empty then return True
    '''
    if text is None:
        raise ValueError("text cannot be None")
    for c in text:
        if c not in HIRAGANA and c not in KATAKANA:
            return False
    return True
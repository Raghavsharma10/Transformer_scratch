def switch_charset(characters, target=''):
    '''
    Transforms an iterable of kana characters to its opposite script.
    For example, it can turn [u'あ', u'い'] into [u'ア', u'イ'],
    or {u'ホ': u'ボ} into {u'ほ': u'ぼ'}.

    There are no safety checks--keep in mind that the correct source and target
    values must be set, otherwise the resulting characters will be garbled.
    '''
    if isinstance(characters, dict):
        return _switch_charset_dict(characters, target)
    else:
        return _switch_charset_list(characters, target)
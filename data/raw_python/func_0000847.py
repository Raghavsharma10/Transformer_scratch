def strip_punctuation_space(value):
    "Strip excess whitespace prior to punctuation."
    def strip_punctuation(string):
        replacement_list = (
            (' .',  '.'),
            (' :',  ':'),
            ('( ',  '('),
            (' )',  ')'),
        )
        for match, replacement in replacement_list:
            string = string.replace(match, replacement)
        return string
    if value == None:
        return None
    if type(value) == list:
        return [strip_punctuation(v) for v in value]
    return strip_punctuation(value)
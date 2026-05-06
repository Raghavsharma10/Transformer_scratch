def to_kana(text):
    """
    Use MeCab to turn any text into its phonetic spelling, as katakana
    separated by spaces.
    """
    records = MECAB.analyze(text)
    kana = []
    for record in records:
        if record.pronunciation:
            kana.append(record.pronunciation)
        elif record.reading:
            kana.append(record.reading)
        else:
            kana.append(record.surface)
    return ' '.join(k for k in kana if k)
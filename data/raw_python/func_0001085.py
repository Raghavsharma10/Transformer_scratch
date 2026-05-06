def analyse(content, splitlines=True, format=None, **kwargs):
    ''' Japanese text > tokenize/txt/html '''
    sents = DekoText.parse(content, splitlines=splitlines, **kwargs)
    doc = []
    final = sents
    # Generate output
    if format == 'html':
        for sent in sents:
            doc.append(sent.to_ruby())
        final = '<br/>\n'.join(doc)
    elif format == 'csv':
        for sent in sents:
            doc.append('\n'.join([x.to_csv() for x in sent]) + '\n')
        final = '\n'.join(doc)
    elif format == 'txt':
        final = '\n'.join([str(x) for x in sents])
    return final
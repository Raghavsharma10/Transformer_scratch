def splitter_support(py2enc):
    '''Create tokenizer for use in boundary constraint parsing.

    :param py2enc: Encoding used by Python 2 environment.
    :type py2enc: str
    '''
    if sys.version < '3':
        def _fn_sentence(pattern, sentence):
            if REGEXTYPE == type(pattern):
                if pattern.flags & re.UNICODE:
                    return sentence.decode(py2enc)
                else:
                    return sentence
            else:
                return sentence
        def _fn_token2str(pattern):
            if REGEXTYPE == type(pattern):
                if pattern.flags & re.UNICODE:
                    def _fn(token):
                        return token.encode(py2enc)
                else:
                    def _fn(token):
                        return token
            else:
                def _fn(token):
                    return token
            return _fn
    else:
        def _fn_sentence(pattern, sentence):
            return sentence
        def _fn_token2str(pattern):
            def _fn(token):
                return token
            return _fn

    def _fn_tokenize_pattern(text, pattern):
        pos = 0
        sentence = _fn_sentence(pattern, text)
        postprocess = _fn_token2str(pattern)
        for m in re.finditer(pattern, sentence):
            if pos < m.start():
                token = postprocess(sentence[pos:m.start()])
                yield (token.strip(), False)
                pos = m.start()
            token = postprocess(sentence[pos:m.end()])
            yield (token.strip(), True)
            pos = m.end()
        if pos < len(sentence):
            token = postprocess(sentence[pos:])
            yield (token.strip(), False)

    def _fn_tokenize_features(text, features):
        acc = []
        acc.append((text.strip(), False))

        for feat in features:
            for i,e in enumerate(acc):
                if e[1]==False:
                    tmp = list(_fn_tokenize_pattern(e[0], feat))
                    if len(tmp) > 0:
                        acc.pop(i)
                        acc[i:i] = tmp
        return acc
                        
    return _fn_tokenize_pattern, _fn_tokenize_features
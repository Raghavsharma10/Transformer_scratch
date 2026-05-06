def tokenize_sent(mtokens, raw='', auto_strip=True):
    ''' Tokenize a text to multiple sentences '''
    sents = []
    bucket = []
    cfrom = 0
    cto = 0
    token_cfrom = 0
    logger = getLogger()
    logger.debug("raw text: {}".format(raw))
    logger.debug("tokens: {}".format(mtokens))
    for t in mtokens:
        if t.is_eos:
            continue
        token_cfrom = raw.find(t.surface, cto)
        cto = token_cfrom + len(t.surface)  # also token_cto
        logger.debug("processing token {} <{}:{}>".format(t, token_cfrom, cto))
        bucket.append(t)
        # sentence ending
        if t.pos == '記号' and t.sc1 == '句点':
            sent_text = raw[cfrom:cto]
            getLogger().debug("sent_text = {} | <{}:{}>".format(sent_text, cfrom, cto))
            if auto_strip:
                sent_text = sent_text.strip()
            sents.append(MeCabSent(sent_text, bucket))
            logger.debug("Found a sentence: {}".format(sent_text))
            cfrom = cto
            bucket = []
    if bucket:
        logger.debug("Bucket is not empty: {}".format(bucket))
        sent_text = raw[cfrom:cto]
        logger.debug("remaining text: {} [{}:{}]".format(sent_text, cfrom, cto))
        if auto_strip:
            sent_text = sent_text.strip()
        sents.append(MeCabSent(sent_text, bucket))
    return sents
def get_ranked_english():
    '''
    wikitionary has a list of ~40k English words, ranked by frequency of occurance in TV and movie transcripts.
    more details at:
    http://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/TV/2006/explanation

    the list is separated into pages of 1000 or 2000 terms each.
    * the first 10k words are separated into pages of 1000 terms each.
    * the remainder is separated into pages of 2000 terms each:
    '''
    URL_TMPL = 'http://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/TV/2006/%s'
    urls = []
    for i in range(10):
        freq_range = "%d-%d" % (i * 1000 + 1, (i+1) * 1000)
        urls.append(URL_TMPL % freq_range)

    for i in range(0,15):
        freq_range = "%d-%d" % (10000 + 2 * i * 1000 + 1, 10000 + (2 * i + 2) * 1000)
        urls.append(URL_TMPL % freq_range)

    urls.append(URL_TMPL % '40001-41284')

    ranked_terms = [] # ordered by rank, in decreasing frequency.
    for url in urls:
        html, is_cached = wiki_download(url)
        if not is_cached:
            time.sleep(SLEEP_TIME)
        new_terms = parse_wiki_terms(html)
        ranked_terms.extend(new_terms)

    return ranked_terms
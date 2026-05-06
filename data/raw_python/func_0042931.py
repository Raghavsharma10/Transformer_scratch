def parse_wiki_terms(doc):
    '''who needs an html parser. fragile hax, but checks the result at the end'''
    results = []
    last3 = ['', '', '']
    header = True
    for line in doc.split('\n'):
        last3.pop(0)
        last3.append(line.strip())
        if all(s.startswith('<td>') and not s == '<td></td>' for s in last3):
            if header:
                header = False
                continue
            last3 = [s.replace('<td>', '').replace('</td>', '').strip() for s in last3]
            rank, term, count = last3
            rank = int(rank.split()[0])
            term = term.replace('</a>', '')
            term = term[term.index('>')+1:].lower()
            results.append(term)
    assert len(results) in [1000, 2000, 1284] # early docs have 1k entries, later have 2k, last doc has 1284
    return results
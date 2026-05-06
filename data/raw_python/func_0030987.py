def get_current_ontology_date():
    """Get the release date of the current Gene Ontolgo release."""
    with closing(requests.get(
            'http://geneontology.org/ontology/go-basic.obo',
            stream=True)) as r:
        for i, l in enumerate(r.iter_lines(decode_unicode=True)):
            if i == 1:
                assert l.split(':')[0] == 'data-version'
                date = l.split('/')[-1]
                break

    return date
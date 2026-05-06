def anon():
    '''Check for candidates to anonymization'''
    header(anon.__doc__)
    filename = 'urls_to_check.csv'

    candidates = Advice.objects(__raw__={
        '$or': [
            {'subject': {
                '$regex': '(Monsieur|Madame|Docteur|Mademoiselle)\s+[^X\s\.]{3}',
                '$options': 'imx',
            }},
            {'content': {
                '$regex': '(Monsieur|Madame|Docteur|Mademoiselle)\s+[^X\s\.]{3}',
                '$options': 'imx',
            }}
        ]
    })

    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        # Generate header
        writer.writerow(csv.ANON_HEADER)

        for idx, advice in enumerate(candidates, 1):
            writer.writerow(csv.to_anon_row(advice))
            echo('.' if idx % 50 else white(idx), nl=False)
        echo(white(idx) if idx % 50 else '')

    success('Total: {0} candidates', len(candidates))
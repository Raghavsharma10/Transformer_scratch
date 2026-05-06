def load(patterns, full_reindex):
    '''
    Load one or more CADA CSV files matching patterns
    '''
    header('Loading CSV files')
    for pattern in patterns:
        for filename in iglob(pattern):
            echo('Loading {}'.format(white(filename)))
            with open(filename) as f:
                reader = csv.reader(f)
                # Skip header
                reader.next()
                for idx, row in enumerate(reader, 1):
                    try:
                        advice = csv.from_row(row)
                        skipped = False
                        if not full_reindex:
                            index(advice)
                        echo('.' if idx % 50 else white(idx), nl=False)
                    except Exception:
                        echo(cyan('s') if idx % 50 else white('{0}(s)'.format(idx)), nl=False)
                        skipped = True
                if skipped:
                    echo(white('{}(s)'.format(idx)) if idx % 50 else '')
                else:
                    echo(white(idx) if idx % 50 else '')
                success('Processed {0} rows'.format(idx))
    if full_reindex:
        reindex()
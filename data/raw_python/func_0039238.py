def reindex():
    '''Reindex all advices'''
    header('Reindexing all advices')
    echo('Deleting index {0}', white(es.index_name))
    if es.indices.exists(es.index_name):
        es.indices.delete(index=es.index_name)
    es.initialize()

    idx = 0
    for idx, advice in enumerate(Advice.objects, 1):
        index(advice)
        echo('.' if idx % 50 else white(idx), nl=False)
    echo(white(idx) if idx % 50 else '')
    es.indices.refresh(index=es.index_name)
    success('Indexed {0} advices', idx)
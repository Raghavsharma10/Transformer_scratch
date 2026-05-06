def build_mp(b, stage, source_name, force):
    """Ingest a source, using only arguments that can be pickled, for multiprocessing access"""

    source = b.source(source_name)

    with b.progress.start('build_mp',stage,message="MP build", source=source) as ps:
        ps.add(message='Running source {}'.format(source.name), source=source, state='running')
        r = b.build_source(stage, source, ps, force)

    return r
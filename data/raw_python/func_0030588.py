def ingest_mp(b, stage, source_name, clean_files):
    """Ingest a source, using only arguments that can be pickled, for multiprocessing access"""

    source = b.source(source_name)

    with b.progress.start('ingest_mp',0,message="MP ingestion", source=source) as ps:
        r =  b._ingest_source(source, ps, clean_files)

    return r
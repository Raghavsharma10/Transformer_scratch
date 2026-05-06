def uclust_fasta_sort_from_filepath(
        fasta_filepath,
        output_filepath=None,
        tmp_dir=gettempdir(),
        HALT_EXEC=False):
    """Generates sorted fasta file via uclust --mergesort."""
    if not output_filepath:
        _, output_filepath = mkstemp(dir=tmp_dir, prefix='uclust_fasta_sort',
                                     suffix='.fasta')

    app = Uclust(params={'--tmpdir': tmp_dir},
                 TmpDir=tmp_dir, HALT_EXEC=HALT_EXEC)

    app_result = app(data={'--mergesort': fasta_filepath,
                           '--output': output_filepath})

    return app_result
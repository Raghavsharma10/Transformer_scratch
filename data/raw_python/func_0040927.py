def find_genus(files, database, threads=12):
    """
    Uses MASH to find the genus of fasta files.
    :param files: File dictionary returned by filer method.
    :param database: Path to reduced refseq database sketch.
    :param threads: Number of threads to run mash with.
    :return: genus_dict: Dictionary of genus for each sample. Will return NA if genus could not be found.
    """
    genus_dict = dict()
    tmpdir = str(time.time()).split('.')[-1]
    if not os.path.isdir(tmpdir):
        os.makedirs(tmpdir)
    for file_name, fasta in files.items():
        mash.screen(database, fasta,
                    threads=threads,
                    w='',
                    i=0.95,
                    output_file=os.path.join(tmpdir, 'screen.tab'))
        screen_output = mash.read_mash_screen(os.path.join(tmpdir, 'screen.tab'))
        try:
            os.remove(os.path.join(tmpdir, 'screen.tab'))
        except IOError:
            pass
        try:
            genus = screen_output[0].query_id.split('/')[-3]
            if genus == 'Shigella':
                genus = 'Escherichia'
            genus_dict[file_name] = genus
        except IndexError:
            genus_dict[file_name] = 'NA'

    shutil.rmtree(tmpdir)
    return genus_dict
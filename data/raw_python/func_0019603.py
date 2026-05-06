def download_annotation(genomebuild, gene_file): 
    """
    Download gene annotation from UCSC based on genomebuild.

    Will check UCSC, Ensembl and RefSeq annotation.

    Parameters
    ----------
    genomebuild : str
        UCSC genome name.
    gene_file : str
        Output file name.
    """
    pred_bin = "genePredToBed"
    pred = find_executable(pred_bin)
    if not pred:
        sys.stderr.write("{} not found in path!\n".format(pred_bin))
        sys.exit(1)

    tmp = NamedTemporaryFile(delete=False, suffix=".gz")

    anno = []
    f = urlopen(UCSC_GENE_URL.format(genomebuild))
    p = re.compile(r'\w+.Gene.txt.gz')
    for line in f.readlines():
        m = p.search(line.decode())
        if m:
            anno.append(m.group(0))

    sys.stderr.write("Retrieving gene annotation for {}\n".format(genomebuild))
    url = ""
    for a in ANNOS:
        if a in anno:
            url = UCSC_GENE_URL.format(genomebuild) + a
            break
    if url:
        sys.stderr.write("Using {}\n".format(url))
        urlretrieve(
                url,
                tmp.name
                )
         
        with gzip.open(tmp.name) as f:
            cols = f.readline().decode(errors='ignore').split("\t")

        start_col = 1
        for i,col in enumerate(cols):
            if col == "+" or col == "-":
                start_col = i - 1
                break
        end_col = start_col + 10
       
        cmd = "zcat {} | cut -f{}-{} | {} /dev/stdin {}"
        print(cmd.format(tmp.name, start_col, end_col, pred, gene_file))
        sp.call(cmd.format(
            tmp.name, start_col, end_col, pred, gene_file), 
            shell=True)

    else:
        sys.stderr.write("No annotation found!")
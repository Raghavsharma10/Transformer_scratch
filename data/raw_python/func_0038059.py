def write_tex():
    """
    Finds all of the output data files, and writes them out to .tex
    """
    datadir = livvkit.index_dir
    outdir = os.path.join(datadir, "tex")
    print(outdir)
    # functions.mkdir_p(outdir)

    data_files = glob.glob(datadir + "/**/*.json", recursive=True)

    for each in data_files:
        data = functions.read_json(each)
        tex = translate_page(data)
        outfile = os.path.join(outdir, os.path.basename(each).replace('json', 'tex'))
        with open(outfile, 'w') as f:
            f.write(tex)
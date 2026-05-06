def generate_simple_vcf(filename, variant_collection):
    """
    Output a very simple metadata-free VCF for each variant in a variant_collection.
    """
    contigs = []
    positions = []
    refs = []
    alts = []
    for variant in variant_collection:
        contigs.append("chr" + variant.contig)
        positions.append(variant.start)
        refs.append(variant.ref)
        alts.append(variant.alt)
    df = pd.DataFrame()
    df["contig"] = contigs
    df["position"] = positions
    df["id"] = ["."] * len(variant_collection)
    df["ref"] = refs
    df["alt"] = alts
    df["qual"] = ["."] * len(variant_collection)
    df["filter"] = ["."] * len(variant_collection)
    df["info"] = ["."] * len(variant_collection)
    df["format"] = ["GT:AD:DP"] * len(variant_collection)
    normal_ref_depths = [randint(1, 10) for v in variant_collection]
    normal_alt_depths = [randint(1, 10) for v in variant_collection]
    df["n1"] = ["0:%d,%d:%d" % (normal_ref_depths[i], normal_alt_depths[i],
                                normal_ref_depths[i] + normal_alt_depths[i])
                for i in range(len(variant_collection))]
    tumor_ref_depths = [randint(1, 10) for v in variant_collection]
    tumor_alt_depths = [randint(1, 10) for v in variant_collection]
    df["t1"] = ["0/1:%d,%d:%d" % (tumor_ref_depths[i], tumor_alt_depths[i], tumor_ref_depths[i] + tumor_alt_depths[i])
                for i in range(len(variant_collection))]

    with open(filename, "w") as f:
        f.write("##fileformat=VCFv4.1\n")
        f.write("##reference=file:///projects/ngs/resources/gatk/2.3/ucsc.hg19.fasta\n")

    with open(filename, "a") as f:
        df.to_csv(f, sep="\t", index=None, header=None)
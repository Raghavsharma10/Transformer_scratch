def load_ensembl_coverage(cohort, coverage_path, min_tumor_depth, min_normal_depth=0,
                          pageant_dir_fn=None):
    """
    Load in Pageant CoverageDepth results with Ensembl loci.

    coverage_path is a path to Pageant CoverageDepth output directory, with
    one subdirectory per patient and a `cdf.csv` file inside each patient subdir.

    If min_normal_depth is 0, calculate tumor coverage. Otherwise, calculate
    join tumor/normal coverage.

    pageant_dir_fn is a function that takes in a Patient and produces a Pageant
    dir name.

    Last tested with Pageant CoverageDepth version 1ca9ed2.
    """
    # Function to grab the pageant file name using the Patient
    if pageant_dir_fn is None:
        pageant_dir_fn = lambda patient: patient.id

    columns_both = [
        "depth1", # Normal
        "depth2", # Tumor
        "onBP1",
        "onBP2",
        "numOnLoci",
        "fracBPOn1",
        "fracBPOn2",
        "fracLociOn",
        "offBP1",
        "offBP2",
        "numOffLoci",
        "fracBPOff1",
        "fracBPOff2",
        "fracLociOff",
    ]
    columns_single = [
        "depth",
        "onBP",
        "numOnLoci",
        "fracBPOn",
        "fracLociOn",
        "offBP",
        "numOffLoci",
        "fracBPOff",
        "fracLociOff"
    ]
    if min_normal_depth < 0:
        raise ValueError("min_normal_depth must be >= 0")
    use_tumor_only = (min_normal_depth == 0)
    columns = columns_single if use_tumor_only else columns_both
    ensembl_loci_dfs = []
    for patient in cohort:
        patient_ensembl_loci_df = pd.read_csv(
            path.join(coverage_path, pageant_dir_fn(patient), "cdf.csv"),
            names=columns,
            header=1)
        # pylint: disable=no-member
        # pylint gets confused by read_csv
        if use_tumor_only:
            depth_mask = (patient_ensembl_loci_df.depth == min_tumor_depth)
        else:
            depth_mask = (
                (patient_ensembl_loci_df.depth1 == min_normal_depth) &
                (patient_ensembl_loci_df.depth2 == min_tumor_depth))
        patient_ensembl_loci_df = patient_ensembl_loci_df[depth_mask]
        assert len(patient_ensembl_loci_df) == 1, (
            "Incorrect number of tumor={}, normal={} depth loci results: {} for patient {}".format(
                min_tumor_depth, min_normal_depth, len(patient_ensembl_loci_df), patient))
        patient_ensembl_loci_df["patient_id"] = patient.id
        ensembl_loci_dfs.append(patient_ensembl_loci_df)
    ensembl_loci_df = pd.concat(ensembl_loci_dfs)
    ensembl_loci_df["MB"] = ensembl_loci_df.numOnLoci / 1000000.0
    return ensembl_loci_df[["patient_id", "numOnLoci", "MB"]]
def random_cohort(size, cache_dir, data_dir=None,
                  min_random_variants=None,
                  max_random_variants=None,
                  seed_val=1234):
    """
    Parameters
    ----------
    min_random_variants: optional, int
        Minimum number of random variants to be generated per patient.
    max_random_variants: optional, int
        Maximum number of random variants to be generated per patient.
    """
    seed(seed_val)
    d = {}
    d["id"] = [str(id) for id in range(size)]
    d["age"] = choice([10, 15, 28, 32, 59, 62, 64, 66, 68], size)
    d["smoker"] = choice([False, True], size)
    d["OS"] = [randint(10, 1000) for i in range(size)]
    d["PFS"] = [int(os * 0.6) for os in d["OS"]]
    d["benefit"] = [pfs < 50 for pfs in d["PFS"]]
    d["random"] = [randint(100) for i in range(size)]
    d["random_boolean"] = choice([False, True], size)
    d["benefit_correlate"] = [randint(50) if benefit else randint(20) for benefit in d["benefit"]]
    d["benefit_correlate_boolean"] = [True if corr > 10 else False for corr in d["benefit_correlate"]]
    d["deceased"] = choice([False, True], size)
    d["progressed_or_deceased"] = [deceased or choice([False, True]) for deceased in d["deceased"]]
    df = pd.DataFrame(d)
    patients = []
    for i, row in df.iterrows():
        snv_vcf_paths = None
        if max_random_variants is not None and min_random_variants is not None:
            if data_dir is None:
                raise ValueError("data_dir must be provided if random variants are being generated.")
            vcf_path = path.join(data_dir, "patient_%s_mutect.vcf" % row["id"])
            generate_simple_vcf(
                vcf_path, generate_random_missense_variants(num_variants=randint(min_random_variants, max_random_variants)))
            snv_vcf_paths = [vcf_path]
        patient = Patient(
            id=row["id"],
            os=row["OS"],
            pfs=row["PFS"],
            benefit=row["benefit"],
            deceased=row["deceased"],
            progressed_or_deceased=row["progressed_or_deceased"],
            hla_alleles=["HLA-A02:01"],
            variants={"snv": snv_vcf_paths},
            additional_data=row)
        patients.append(patient)
    return Cohort(
        patients=patients,
        cache_dir=cache_dir,
        mhc_class=RandomBindingPredictor)
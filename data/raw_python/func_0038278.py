def sam2rnf(args):
    """Convert SAM to RNF-based FASTQ with respect to argparse parameters.

	Args:
		args (...): Arguments parsed by argparse
	"""

    rnftools.mishmash.Source.recode_sam_reads(
        sam_fn=args.sam_fn,
        fastq_rnf_fo=args.fq_fo,
        fai_fo=args.fai_fo,
        genome_id=args.genome_id,
        number_of_read_tuples=10**9,
        simulator_name=args.simulator_name,
        allow_unmapped=args.allow_unmapped,
    )
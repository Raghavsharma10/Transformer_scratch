def main():
    """
    This is a Toil pipeline for the UNC best practice RNA-Seq analysis.
    RNA-seq fastqs are combined, aligned, sorted, filtered, and quantified.

    Please read the README.md located in the same directory.
    """
    # Define Parser object and add to toil
    parser = build_parser()
    Job.Runner.addToilOptions(parser)
    args = parser.parse_args()
    # Store inputs from argparse
    inputs = {'config': args.config,
              'config_fastq': args.config_fastq,
              'input': args.input,
              'unc.bed': args.unc,
              'hg19.transcripts.fa': args.fasta,
              'composite_exons.bed': args.composite_exons,
              'normalize.pl': args.normalize,
              'output_dir': args.output_dir,
              'rsem_ref.zip': args.rsem_ref,
              'chromosomes.zip': args.chromosomes,
              'ebwt.zip': args.ebwt,
              'ssec': args.ssec,
              's3_dir': args.s3_dir,
              'sudo': args.sudo,
              'single_end_reads': args.single_end_reads,
              'upload_bam_to_s3': args.upload_bam_to_s3,
              'uuid': None,
              'sample.tar': None,
              'cpu_count': None}

    # Launch jobs
    Job.Runner.startToil(Job.wrapJobFn(download_shared_files, inputs), args)
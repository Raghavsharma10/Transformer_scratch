def generate_index(credentials, instance_config, instance_name,
                   script_dir, genome_file, output_dir, annotation_file=None,
                   splice_overhang=100,
                   num_threads=8, chromosome_bin_bits=18,
                   genome_memory_limit=31000000000,
                   self_destruct=True,
                   **kwargs):
    """Generates a STAR index.

    Recommended machine type: "n1-highmem-8" (52 GB of RAM, 8 vCPUs)

    TODO: docstring"""

    template = _TEMPLATE_ENV.get_template(
        os.path.join('generate_index.sh'))
    startup_script = template.render(
        script_dir=script_dir,
        genome_file=genome_file,
        annotation_file=annotation_file,
        splice_overhang=splice_overhang,
        output_dir=output_dir,
        num_threads=num_threads,
        chromosome_bin_bits=chromosome_bin_bits,
        genome_memory_limit=genome_memory_limit,
        self_destruct=self_destruct)

    op_name = instance_config.create_instance(
        credentials, instance_name, startup_script=startup_script, **kwargs)

    return op_name
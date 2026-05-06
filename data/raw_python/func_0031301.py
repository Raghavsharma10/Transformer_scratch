def sra_download_paired_end(credentials, instance_config, instance_name,
                            script_dir, sra_run_acc, output_dir, **kwargs):    
    """Download paired-end reads from SRA and convert to gzip'ed FASTQ files.

    TODO: docstring"""

    template = _TEMPLATE_ENV.get_template('sra_download_paired-end.sh')
    startup_script = template.render(
        script_dir=script_dir,
        sra_run_acc=sra_run_acc,
        output_dir=output_dir)

    if len(startup_script) > 32768:
        raise ValueError('Startup script larger than 32,768 bytes!')

    #print(startup_script)

    instance_config.create_instance(
        credentials, instance_name, startup_script=startup_script, **kwargs)
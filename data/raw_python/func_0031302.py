def trim_fastq(credentials, instance_config, instance_name,
               script_dir, input_file, output_file,
               trim_crop, trim_headcrop=0, self_destruct=True,
               **kwargs):
    """Trims a FASTQ file.
    
    TODO: docstring"""
    template = _TEMPLATE_ENV.get_template('trim_fastq.sh')
    startup_script = template.render(
        script_dir=script_dir,
        input_file=input_file,
        output_file=output_file,
        trim_crop=trim_crop,
        trim_headcrop=trim_headcrop,
        self_destruct=self_destruct)

    if len(startup_script) > 32768:
        raise ValueError('Startup script larger than 32,768 bytes!')

    #print(startup_script)

    instance_config.create_instance(
        credentials, instance_name, startup_script=startup_script, **kwargs)
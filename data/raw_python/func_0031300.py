def run_fastqc(credentials, instance_config, instance_name,
               script_dir, input_file, output_dir, self_destruct=True,
               **kwargs):
    """Run FASTQC.

    TODO: docstring"""
    template = _TEMPLATE_ENV.get_template('fastqc.sh')
    startup_script = template.render(
        script_dir=script_dir,
        input_file=input_file,
        output_dir=output_dir,
        self_destruct=self_destruct)

    if len(startup_script) > 32768:
        raise ValueError('Startup script larger than 32,768 bytes!')

    #print(startup_script)

    op_name = instance_config.create_instance(
        credentials, instance_name, startup_script=startup_script, **kwargs)

    return op_name
def verify(conf, input_requirements_filename):
    """Verifying that given requirements file is not missing any pins

    args:

    input_requirements_filename: requriements file to verify

    """
    exit_if_file_not_exists(input_requirements_filename, conf)

    cireqs.check_if_requirements_are_up_to_date(
        requirements_filename=input_requirements_filename,
        **conf._asdict())
    click.echo(click.style('✓', fg='green') + " {} has been verified".format(input_requirements_filename))
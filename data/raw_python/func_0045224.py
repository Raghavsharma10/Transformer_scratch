def expand(conf, output_requirements_filename, input_requirements_filename):
    """Expand given requirements file by extending it using pip freeze

    args:

    input_requirements_filename: the requirements filename to expand

    output_requirements_filename: the output filename for the expanded
    requirements file
    """
    exit_if_file_not_exists(input_requirements_filename, conf)
    cireqs.expand_requirements(
        requirements_filename=input_requirements_filename,
        expanded_requirements_filename=output_requirements_filename,
        **conf._asdict()
    )
    click.echo(click.style('✓', fg='green') + " {} has been expanded into {}".format(
        input_requirements_filename, output_requirements_filename
    ))
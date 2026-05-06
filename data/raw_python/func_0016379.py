def show(cobertura_file, format, output, source, source_prefix):
    """show coverage summary of a Cobertura report"""
    cobertura = Cobertura(cobertura_file, source=source)
    Reporter = reporters[format]
    reporter = Reporter(cobertura)
    report = reporter.generate()

    if not isinstance(report, bytes):
        report = report.encode('utf-8')

    isatty = True if output is None else output.isatty()
    click.echo(report, file=output, nl=isatty)
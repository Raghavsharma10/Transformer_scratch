def diff(
        cobertura_file1, cobertura_file2,
        color, format, output, source1, source2,
        source_prefix1, source_prefix2, source):
    """compare coverage of two Cobertura reports"""
    cobertura1 = Cobertura(
        cobertura_file1,
        source=source1,
        source_prefix=source_prefix1
    )
    cobertura2 = Cobertura(
        cobertura_file2,
        source=source2,
        source_prefix=source_prefix2
    )

    Reporter = delta_reporters[format]
    reporter_args = [cobertura1, cobertura2]
    reporter_kwargs = {'show_source': source}

    isatty = True if output is None else output.isatty()

    if format == 'text':
        color = isatty if color is None else color is True
        reporter_kwargs['color'] = color

    reporter = Reporter(*reporter_args, **reporter_kwargs)
    report = reporter.generate()

    if not isinstance(report, bytes):
        report = report.encode('utf-8')

    click.echo(report, file=output, nl=isatty, color=color)

    exit_code = get_exit_code(reporter.differ, source)
    raise SystemExit(exit_code)
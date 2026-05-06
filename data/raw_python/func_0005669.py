def init(quick):
    # type: () -> None
    """ Create an empty pelconf.yaml from template """
    config_file = 'pelconf.yaml'
    prompt = "-- <35>{} <32>already exists. Wipe it?<0>".format(config_file)

    if exists(config_file) and not click.confirm(shell.fmt(prompt)):
        log.info("Canceled")
        return

    form = InitForm().run(quick=quick)

    log.info('Writing <35>{}'.format(config_file))
    pelconf_template = conf.load_template('pelconf.yaml')
    fs.write_file(config_file, pelconf_template.format(**form.values))
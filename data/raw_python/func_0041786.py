def _compile_qt_resources():
    """
    Compiles PyQT resources file
    """
    if config.QT_RES_SRC():
        epab.utils.ensure_exe('pyrcc5')
        LOGGER.info('compiling Qt resources')
        elib_run.run(f'pyrcc5 {config.QT_RES_SRC()} -o {config.QT_RES_TGT()}')
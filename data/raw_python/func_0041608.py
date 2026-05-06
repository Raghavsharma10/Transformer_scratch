def setConnStringForWindows():
    """ Set Conn String for Windiws

    Windows has a different way of forking processes, which causes the
    @worker_process_init.connect signal not to work in "CeleryDbConnInit"


    """
    global _dbConnectString
    from peek_platform.file_config.PeekFileConfigABC import PeekFileConfigABC
    from peek_platform.file_config.PeekFileConfigSqlAlchemyMixin import \
        PeekFileConfigSqlAlchemyMixin
    from peek_platform import PeekPlatformConfig

    class _WorkerTaskConfigMixin(PeekFileConfigABC,
                           PeekFileConfigSqlAlchemyMixin):
        pass

    PeekPlatformConfig.componentName = peekWorkerName

    _dbConnectString = _WorkerTaskConfigMixin().dbConnectString
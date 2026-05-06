def check_name_not_on_dapi(cls, dap):
        '''Check that the package_name is not registered on Dapi.

        Return list of problems.'''
        problems = list()

        if dap.meta['package_name']:
            from . import dapicli
            d = dapicli.metadap(dap.meta['package_name'])
            if d:
                problems.append(DapProblem('This dap name is already registered on Dapi',
                                           level=logging.WARNING))
        return problems
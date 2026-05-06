def check_topdir(cls, dap):
        '''Check that everything is in the correct top-level directory.

        Return a list of DapProblems'''
        problems = list()
        dirname = os.path.dirname(dap._meta_location)

        if not dirname:
            msg = 'meta.yaml is not in top-level directory'
            problems.append(DapProblem(msg))

        else:
            for path in dap.files:
                if not path.startswith(dirname):
                    msg = path + ' is outside of ' + dirname + ' top-level directory'
                    problems.append(DapProblem(msg))

        if dap.meta['package_name'] and dap.meta['version']:
            desired_dirname = dap._dirname()
            desired_filename = desired_dirname + '.dap'

            if dirname and dirname != desired_dirname:
                msg = 'Top-level directory with meta.yaml is not named ' + desired_dirname
                problems.append(DapProblem(msg))

            if dap.basename != desired_filename:
                msg = 'The dap filename is not ' + desired_filename
                problems.append(DapProblem(msg))

        return problems
def check_meta(cls, dap):
        '''Check the meta.yaml in the dap.

        Return a list of DapProblems.'''
        problems = list()

        # Check for non array-like metadata
        for datatype in (Dap._required_meta | Dap._optional_meta) - Dap._array_meta:
            if not dap._isvalid(datatype):
                msg = datatype + ' is not valid (or required and unspecified)'
                problems.append(DapProblem(msg))

        # Check for the array-like metadata
        for datatype in Dap._array_meta:
            ok, bads = dap._arevalid(datatype)
            if not ok:
                if not bads:
                    msg = datatype + ' is not a valid non-empty list'
                    problems.append(DapProblem(msg))
                else:
                    for bad in bads:
                        msg = bad + ' in ' + datatype + ' is not valid or is a duplicate'
                        problems.append(DapProblem(msg))

        # Check that there is no unknown metadata
        leftovers = set(dap.meta.keys()) - (Dap._required_meta | Dap._optional_meta)
        if leftovers:
            msg = 'Unknown metadata: ' + str(leftovers)
            problems.append(DapProblem(msg))

        # Check that package_name is not longer than 200 characters
        if len(dap.meta.get('package_name', '')) > 200:
            msg = 'Package name is too long. It must not exceed 200 characters.'
            problems.append(DapProblem(msg))

        return problems
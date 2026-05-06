def check_no_self_dependency(cls, dap):
        '''Check that the package does not depend on itself.

        Return a list of problems.'''
        problems = list()

        if 'package_name' in dap.meta and 'dependencies' in dap.meta:
            dependencies = set()

            for dependency in dap.meta['dependencies']:
                if 'dependencies' in dap._badmeta and dependency in dap._badmeta['dependencies']:
                    continue

                # No version specified
                if not re.search(r'[<=>]', dependency):
                    dependencies.add(dependency)

                # Version specified
                for mark in ['==', '>=', '<=', '<', '>']:
                    dep = dependency.split(mark)
                    if len(dep) == 2:
                        dependencies.add(dep[0].strip())
                        break

            if dap.meta['package_name'] in dependencies:
                msg = 'Depends on dap with the same name as itself'
                problems.append(DapProblem(msg))

        return problems
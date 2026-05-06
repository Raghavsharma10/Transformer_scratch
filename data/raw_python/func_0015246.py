def check_files(cls, dap):
        '''Check that there are only those files the standard accepts.

        Return list of DapProblems.'''
        problems = list()
        dirname = os.path.dirname(dap._meta_location)

        if dirname:
            dirname += '/'
        files = [f for f in dap.files if f.startswith(dirname)]
        if len(files) == 1:
            msg = 'Only meta.yaml in dap'
            problems.append(DapProblem(msg, level=logging.WARNING))
            return problems

        files.remove(dirname + 'meta.yaml')

        # Report and remove empty directories until no more are found
        emptydirs = dap._get_emptydirs(files)
        while emptydirs:
            for ed in emptydirs:
                msg = ed + ' is empty directory (may be nested)'
                problems.append(DapProblem(msg, logging.WARNING))
                files.remove(ed)
            emptydirs = dap._get_emptydirs(files)

        if dap.meta['package_name']:
            name = dap.meta['package_name']

            dirs = re.compile('^' + dirname + '((assistants(/(crt|twk|prep|extra))?|snippets)(/' +
                              name + ')?|icons(/(crt|twk|prep|extra|snippets)(/' + name +
                              ')?)?|files|(files/(crt|twk|prep|extra|snippets)|doc)(/' + name +
                              '(/.+)?)?)$')
            regs = re.compile('^' + dirname + '((assistants(/(crt|twk|prep|extra))|snippets)/' +
                              name + r'(/[^/]+)?\.yaml|icons/(crt|twk|prep|extra|snippets)/' +
                              name + r'(/[^/]+)?\.(' + Dap._icons_ext +
                              ')|(files/(crt|twk|prep|extra|snippets)|doc)/' + name + '/.+)$')

            to_remove = []
            for f in files:
                if dap._is_dir(f) and not dirs.match(f):
                    msg = f + '/ is not allowed directory'
                    problems.append(DapProblem(msg))
                    to_remove.append(f)
                elif not dap._is_dir(f) and not regs.match(f):
                    msg = f + ' is not allowed file'
                    problems.append(DapProblem(msg))
                    to_remove.append(f)
            for r in to_remove:
                files.remove(r)

            # Subdir yamls need a chief
            for directory in ['assistants/' + t for t in 'crt twk prep extra'.split()] + \
                    ['snippets']:
                prefix = dirname + directory + '/'
                for f in files:
                    if f.startswith(prefix) and dap._is_dir(f) and f + '.yaml' not in files:
                        msg = f + '/ present, but ' + f + '.yaml missing'
                        problems.append(DapProblem(msg))

        # Missing assistants and/or snippets
        if not dap.assistants_and_snippets:
            msg = 'No Assistants or Snippets found'
            problems.append(DapProblem(msg, level=logging.WARNING))

        # Icons
        icons = [dap._strip_leading_dirname(i) for i in dap.icons(strip_ext=True)] # we need to report duplicates
        assistants = set([dap._strip_leading_dirname(a) for a in dap.assistants])  # duplicates are fine here

        duplicates = set([i for i in icons if icons.count(i) > 1])
        for d in duplicates:
            msg = 'Duplicate icon for ' + f
            problems.append(DapProblem(msg, level=logging.WARNING))

        icons = set(icons)
        for i in icons - assistants:
            msg = 'Useless icon for non-exisiting assistant ' + i
            problems.append(DapProblem(msg, level=logging.WARNING))

        for a in assistants - icons:
            msg = 'Missing icon for assistant ' + a
            problems.append(DapProblem(msg, level=logging.WARNING))

        # Source files
        for f in cls._get_files_without_assistants(dap, dirname, files):
            msg = 'Useless files for non-exisiting assistant ' + f
            problems.append(DapProblem(msg, level=logging.WARNING))

        return problems
def fetch_modules(config, relative_path, download_directory):
    """
    Assemble modules which will
    be included in CMakeLists.txt.
    """
    from collections import Iterable, namedtuple, defaultdict
    from autocmake.extract import extract_list, to_d, to_l
    from autocmake.parse_rst import parse_cmake_module

    cleaned_config = defaultdict(lambda: [])

    modules = []
    Module = namedtuple('Module', 'path name')

    num_sources = len(extract_list(config, 'source'))

    print_progress_bar(text='- assembling modules:',
                       done=0,
                       total=num_sources,
                       width=30)

    if 'modules' in config:
        i = 0
        for t in config['modules']:
            for k, v in t.items():

                d = to_d(v)
                for _k, _v in to_d(v).items():
                    cleaned_config[_k] = flat_add(cleaned_config[_k], _v)

                # fetch sources and parse them
                if 'source' in d:
                    for src in to_l(d['source']):
                        i += 1

                        # we download the file
                        module_name = os.path.basename(src)
                        if 'http' in src:
                            path = download_directory
                            name = 'autocmake_{0}'.format(module_name)
                            dst = os.path.join(download_directory, 'autocmake_{0}'.format(module_name))
                            fetch_url(src, dst)
                            file_name = dst
                            fetch_dst_directory = download_directory
                        else:
                            if os.path.exists(src):
                                path = os.path.dirname(src)
                                name = module_name
                                file_name = src
                                fetch_dst_directory = path
                            else:
                                sys.stderr.write("ERROR: {0} does not exist\n".format(src))
                                sys.exit(-1)

                        # we infer config from the module documentation
                        # dictionary d overrides the configuration in the module documentation
                        # this allows to override interpolation inside the module
                        with open(file_name, 'r') as f:
                            parsed_config = parse_cmake_module(f.read(), d)
                            for _k2, _v2 in parsed_config.items():
                                if _k2 not in to_d(v):
                                    # we add to clean_config only if the entry does not exist
                                    # in parent autocmake.yml already
                                    # this allows to override
                                    cleaned_config[_k2] = flat_add(cleaned_config[_k2], _v2)

                        modules.append(Module(path=path, name=name))
                        print_progress_bar(text='- assembling modules:',
                                           done=i,
                                           total=num_sources,
                                           width=30)
        print('')

    return modules, cleaned_config
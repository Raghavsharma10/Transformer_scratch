def main(args=sys.argv):
    """
    Main command-line invocation.
    """
    try:
        opts, args = getopt.gnu_getopt(args[1:], 'p:o:jdt', [
            'jspath=', 'output=', 'private', 'json', 'dependencies', 
            'test', 'help'])
        opts = dict(opts)
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    run_and_exit_if(opts, run_doctests, '--test')
    run_and_exit_if(opts, usage, '--help')

    js_paths = get_path_list(opts)
    docs = CodeBaseDoc(js_paths, '--private' in opts)
    if args:
        selected_files = set(docs.keys()) & set(args)
    else:
        selected_files = list(docs.keys())

    def print_json():
        print(docs.to_json(selected_files))
    run_and_exit_if(opts, print_json, '--json', '-j')

    def print_dependencies():
        for dependency in find_dependencies(selected_files, docs):
            print(dependency)
    run_and_exit_if(opts, print_dependencies, '--dependencies', '-d')

    output = opts.get('--output') or opts.get('-o')
    if output is None and len(args) != 1:
        output = 'apidocs'
    docs.save_docs(selected_files, output)
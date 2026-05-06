def init(options):
    """ Initialize some defaults """

    # Set matlplotlib's backend so LIVVkit can plot to files.
    import matplotlib
    matplotlib.use('agg')

    livvkit.output_dir = os.path.abspath(options.out_dir)
    livvkit.index_dir = livvkit.output_dir
    livvkit.verify = True if options.verify is not None else False
    livvkit.validate = True if options.validate is not None else False
    livvkit.publish = options.publish

    # Get a list of bundles that provide model specific implementations
    available_bundles = [mod for imp, mod, ispkg in pkgutil.iter_modules(bundles.__path__)]

    if options.verify is not None:
        livvkit.model_dir = os.path.normpath(options.verify[0])
        livvkit.bench_dir = os.path.normpath(options.verify[1])
        if not os.path.isdir(livvkit.model_dir):
            print("")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("                       UH OH!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("    Your comparison directory does not exist; please check")
            print("    the path:")
            print("\n"+livvkit.model_dir+"\n\n")
            sys.exit(1)

        if not os.path.isdir(livvkit.bench_dir):
            print("")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("                       UH OH!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("    Your benchmark directory does not exist; please check")
            print("    the path:")
            print("\n"+livvkit.bench_dir+"\n\n")
            sys.exit(1)

        livvkit.model_bundle = os.path.basename(livvkit.model_dir)
        livvkit.bench_bundle = os.path.basename(livvkit.bench_dir)

        if livvkit.model_bundle in available_bundles:
            livvkit.numerics_model_config = os.path.join(
                livvkit.bundle_dir, livvkit.model_bundle, "numerics.json")
            livvkit.numerics_model_module = importlib.import_module(
                ".".join(["livvkit.bundles", livvkit.model_bundle, "numerics"]))

            livvkit.verification_model_config = os.path.join(
                 livvkit.bundle_dir, livvkit.model_bundle, "verification.json")
            livvkit.verification_model_module = importlib.import_module(
                 ".".join(["livvkit.bundles", livvkit.model_bundle, "verification"]))

            livvkit.performance_model_config = os.path.join(
                 livvkit.bundle_dir, livvkit.model_bundle, "performance.json")
            # NOTE: This isn't used right now...
            # livvkit.performance_model_module = importlib.import_module(
            #      ".".join(["livvkit.bundles", livvkit.model_bundle, "performance"]))
        else:
            # TODO: Should implement some error checking here...
            livvkit.verify = False

    if options.validate is not None:
        livvkit.validation_model_configs = options.validate

    if not (livvkit.verify or livvkit.validate) and not options.serve:
        print("")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("                       UH OH!")
        print("----------------------------------------------------------")
        print("    No verification or validation tests found/submitted!")
        print("")
        print("    Use either one or both of the --verify and")
        print("    --validate options to run tests.  For more ")
        print("    information use the --help option, view the README")
        print("    or check https://livvkit.github.io/Docs/")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("")
        sys.exit(1)

    return options
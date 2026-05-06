def run_suite(case, config, summary):
    """ Run the full suite of numerics tests """
    m = importlib.import_module(config['module'])
    m.set_up()
    config["name"] = case
    analysis_data = {}
    bundle = livvkit.numerics_model_module
    model_dir = os.path.join(livvkit.model_dir, config['data_dir'], case)
    bench_dir = os.path.join(livvkit.bench_dir, config['data_dir'], case)
    plot_dir = os.path.join(livvkit.output_dir, "numerics", "imgs")
    config["plot_dir"] = plot_dir
    functions.mkdir_p(plot_dir)
    model_cases = functions.collect_cases(model_dir)
    bench_cases = functions.collect_cases(bench_dir)

    for mscale in sorted(model_cases):
        bscale = bench_cases[mscale] if mscale in bench_cases else []
        for mproc in model_cases[mscale]:
            full_name = '-'.join([mscale, mproc])
            bpath = (os.path.join(bench_dir, mscale, mproc.replace("-", os.path.sep))
                     if mproc in bscale else "")
            mpath = os.path.join(model_dir, mscale, mproc.replace("-", os.path.sep))
            model_data = functions.find_file(mpath, "*" + config["output_ext"])
            bench_data = functions.find_file(bpath, "*" + config["output_ext"])
            analysis_data[full_name] = bundle.get_plot_data(model_data,
                                                            bench_data,
                                                            m.setup[case],
                                                            config)
    try:
        el = m.run(config, analysis_data)
    except KeyError:
        el = elements.error("Numerics Plots", "Missing data")
    result = elements.page(case, config['description'], element_list=el)
    summary[case] = _summarize_result(m, analysis_data, config)
    _print_summary(m, case, summary[case])
    functions.create_page_from_template("numerics.html",
                                        os.path.join(livvkit.index_dir, "numerics", case + ".html"))
    functions.write_json(result, os.path.join(livvkit.output_dir, "numerics"), case + ".json")
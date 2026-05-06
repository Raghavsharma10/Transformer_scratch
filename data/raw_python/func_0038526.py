def run_suite(case, config, summary):
    """ Run the full suite of performance tests """
    config["name"] = case
    timing_data = dict()
    model_dir = os.path.join(livvkit.model_dir, config['data_dir'], case)
    bench_dir = os.path.join(livvkit.bench_dir, config['data_dir'], case)
    plot_dir = os.path.join(livvkit.output_dir, "performance", "imgs")
    model_cases = functions.collect_cases(model_dir)
    bench_cases = functions.collect_cases(bench_dir)
    functions.mkdir_p(plot_dir)

    # Generate all of the timing data
    for subcase in sorted(model_cases):
        bench_subcases = bench_cases[subcase] if subcase in bench_cases else []
        timing_data[subcase] = dict()
        for mcase in model_cases[subcase]:
            config["case"] = "-".join([subcase, mcase])
            bpath = (os.path.join(bench_dir, subcase, mcase.replace("-", os.path.sep))
                     if mcase in bench_subcases else None)
            mpath = os.path.join(model_dir, subcase, mcase.replace("-", os.path.sep))
            timing_data[subcase][mcase] = _analyze_case(mpath, bpath, config)

    # Create scaling and timing breakdown plots
    weak_data = weak_scaling(timing_data, config['scaling_var'],
                             config['weak_scaling_points'])
    strong_data = strong_scaling(timing_data, config['scaling_var'],
                                 config['strong_scaling_points'])

    timing_plots = [
        generate_scaling_plot(weak_data,
                              "Weak scaling for " + case.capitalize(),
                              "runtime (s)", "",
                              os.path.join(plot_dir, case + "_weak_scaling.png")
                              ),
        weak_scaling_efficiency_plot(weak_data,
                                     "Weak scaling efficiency for " + case.capitalize(),
                                     "Parallel efficiency (% of linear)", "",
                                     os.path.join(plot_dir, case + "_weak_scaling_efficiency.png")
                                     ),
        generate_scaling_plot(strong_data,
                              "Strong scaling for " + case.capitalize(),
                              "Runtime (s)", "",
                              os.path.join(plot_dir, case + "_strong_scaling.png")
                              ),
        strong_scaling_efficiency_plot(strong_data,
                                       "Strong scaling efficiency for " + case.capitalize(),
                                       "Parallel efficiency (% of linear)", "",
                                       os.path.join(plot_dir,
                                                    case + "_strong_scaling_efficiency.png")
                                       ),
        ]

    timing_plots = timing_plots + \
        [generate_timing_breakdown_plot(timing_data[s],
                                        config['scaling_var'],
                                        "Timing breakdown for " + case.capitalize()+" "+s,
                                        "",
                                        os.path.join(plot_dir, case+"_"+s+"_timing_breakdown.png")
                                        )
         for s in sorted(six.iterkeys(timing_data), key=functions.sort_scale)]

    # Build an image gallery and write the results
    el = [
            elements.gallery("Performance Plots", timing_plots)
         ]
    result = elements.page(case, config["description"], element_list=el)
    summary[case] = _summarize_result(timing_data, config)
    _print_result(case, summary)
    functions.create_page_from_template("performance.html",
                                        os.path.join(livvkit.index_dir, "performance",
                                                     case + ".html"))
    functions.write_json(result, os.path.join(livvkit.output_dir, "performance"),
                         case + ".json")
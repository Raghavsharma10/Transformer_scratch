def run_suite(case, config, summary):
    """ Run the full suite of verification tests """
    config["name"] = case
    model_dir = os.path.join(livvkit.model_dir, config['data_dir'], case)
    bench_dir = os.path.join(livvkit.bench_dir, config['data_dir'], case)
    tabs = []
    case_summary = LIVVDict()
    model_cases = functions.collect_cases(model_dir)
    bench_cases = functions.collect_cases(bench_dir)

    for subcase in sorted(six.iterkeys(model_cases)):
        bench_subcases = bench_cases[subcase] if subcase in bench_cases else []
        case_sections = []
        for mcase in sorted(model_cases[subcase], key=functions.sort_processor_counts):
            bpath = (os.path.join(bench_dir, subcase, mcase.replace("-", os.path.sep))
                     if mcase in bench_subcases else "")
            mpath = os.path.join(model_dir, subcase, mcase.replace("-", os.path.sep))
            case_result = _analyze_case(mpath, bpath, config)
            case_sections.append(elements.section(mcase, case_result))
            case_summary[subcase] = _summarize_result(case_result,
                                                      case_summary[subcase])
        tabs.append(elements.tab(subcase, section_list=case_sections))

    result = elements.page(case, config["description"], tab_list=tabs)
    summary[case] = case_summary
    _print_summary(case, summary[case])
    functions.create_page_from_template("verification.html",
                                        os.path.join(livvkit.index_dir,
                                                     "verification",
                                                     case + ".html")
                                        )
    functions.write_json(result, os.path.join(livvkit.output_dir, "verification"), case+".json")
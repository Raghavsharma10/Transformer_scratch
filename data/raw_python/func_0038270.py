def run_suite(case, config, summary):
    """ Run the full suite of validation tests """
    m = _load_case_module(case, config)

    result = m.run(case, config)
    summary[case] = _summarize_result(m, result)
    _print_summary(m, case, summary)
   
    if result['Type'] == 'Book':
        for name, page in six.iteritems(result['Data']):
            functions.create_page_from_template("validation.html",
                                                os.path.join(livvkit.index_dir, "validation", name + ".html"))
            functions.write_json(page, os.path.join(livvkit.output_dir, "validation"), name + ".json")
    else:
        functions.create_page_from_template("validation.html",
                                            os.path.join(livvkit.index_dir, "validation", case + ".html"))
        functions.write_json(result, os.path.join(livvkit.output_dir, "validation"), case + ".json")
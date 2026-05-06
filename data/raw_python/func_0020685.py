def generate(self):
    """
    Generates the report
    """
    self._setup()

    header_html = self._generate_header()
    footer_html = self._generate_footer()
    results_topbar_html = self._generate_topbar("results")
    summary_topbar_html = self._generate_topbar("summary")
    logs_topbar_html = self._generate_topbar("logs")
    diff_topbar_html = self._generate_topbar("diff")

    summary_body_html = self._generate_summary_body()
    diff_body_html = self._generate_diff_body()
    summary_html = header_html + summary_topbar_html + summary_body_html + footer_html
    diff_html = header_html + diff_topbar_html + diff_body_html+ footer_html
    Reporter._make_file(summary_html, self.report_info.home_page)
    Reporter._make_file(diff_html,self.report_info.diff_page)

    log_body_html = self._generate_log_body()
    log_html = header_html + logs_topbar_html + log_body_html+footer_html
    Reporter._make_file(log_html, self.report_info.log_page)

    for config_name in self.report_info.config_to_test_names_map.keys():
      config_dir = os.path.join(self.report_info.resource_dir, config_name)
      utils.makedirs(config_dir)

      config_body_html = self._generate_config_body(config_name)
      config_html = header_html + results_topbar_html + config_body_html + footer_html
      config_file = os.path.join(config_dir, config_name + self.report_info.report_file_sfx)
      Reporter._make_file(config_html, config_file)

      for test_name in self.data_source.get_test_names(config_name):
        test_body_html = self._generate_test_body(config_name, test_name)
        test_html = header_html + results_topbar_html + test_body_html + footer_html
        test_file = os.path.join(config_dir, test_name + self.report_info.report_file_sfx)
        Reporter._make_file(test_html, test_file)
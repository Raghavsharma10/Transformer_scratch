def generate(self):
    """
    Generates the report
    """
    self._setup()
    for config_name in self.report_info.config_to_test_names_map.keys():
      config_dir = os.path.join(self.report_info.resource_dir, config_name)
      utils.makedirs(config_dir)
      testsuite = self._generate_junit_xml(config_name)
      with open(os.path.join(self.report_info.junit_xml_path, 'zopkio_junit_reports.xml'), 'w') as file:
          TestSuite.to_file(file, [testsuite], prettyprint=False)
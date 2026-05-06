def get_probes_results(self):
        """Return the results of the RPM probes."""
        probes_results = {}

        probes_results_table = junos_views.junos_rpm_probes_results_table(self.device)
        probes_results_table.get()
        probes_results_items = probes_results_table.items()

        for probe_result in probes_results_items:
            probe_name = py23_compat.text_type(probe_result[0])
            test_results = {
                p[0]: p[1] for p in probe_result[1]
            }
            test_results['last_test_loss'] = napalm_base.helpers.convert(
                int, test_results.pop('last_test_loss'), 0)
            for test_param_name, test_param_value in test_results.items():
                if isinstance(test_param_value, float):
                    test_results[test_param_name] = test_param_value * 1e-3
                    # convert from useconds to mseconds
            test_name = test_results.pop('test_name', '')
            source = test_results.get('source', u'')
            if source is None:
                test_results['source'] = u''
            if probe_name not in probes_results.keys():
                probes_results[probe_name] = {}
            probes_results[probe_name][test_name] = test_results

        return probes_results
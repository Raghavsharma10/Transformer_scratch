def get_probes_config(self):
        """Return the configuration of the RPM probes."""
        probes = {}

        probes_table = junos_views.junos_rpm_probes_config_table(self.device)
        probes_table.get()
        probes_table_items = probes_table.items()

        for probe_test in probes_table_items:
            test_name = py23_compat.text_type(probe_test[0])
            test_details = {
                p[0]: p[1] for p in probe_test[1]
            }
            probe_name = napalm_base.helpers.convert(
                py23_compat.text_type, test_details.pop('probe_name'))
            target = napalm_base.helpers.convert(
                py23_compat.text_type, test_details.pop('target', ''))
            test_interval = napalm_base.helpers.convert(int, test_details.pop('test_interval', '0'))
            probe_count = napalm_base.helpers.convert(int, test_details.pop('probe_count', '0'))
            probe_type = napalm_base.helpers.convert(
                py23_compat.text_type, test_details.pop('probe_type', ''))
            source = napalm_base.helpers.convert(
                py23_compat.text_type, test_details.pop('source_address', ''))
            if probe_name not in probes.keys():
                probes[probe_name] = {}
            probes[probe_name][test_name] = {
                'probe_type': probe_type,
                'target': target,
                'source': source,
                'probe_count': probe_count,
                'test_interval': test_interval
            }

        return probes
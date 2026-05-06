def run_check(self, template_name=None, service_dir=None):
        " Run checking scripts. "

        print_header('Check requirements', sep='-')
        map(lambda cmd: call("bash %s" % cmd), self._gen_scripts(
            'check', template_name=template_name, service_dir=service_dir))
        return True
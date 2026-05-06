def action(self, includes: dict, variables: dict) -> tuple:
        """
        Call external script.

        :param includes: testcase's includes
        :param variables: variables
        :return: script's output
        """
        json_args = fill_template_str(json.dumps(self.data), variables)
        p = subprocess.Popen([self.module, json_args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if p.wait() == 0:
            out = p.stdout.read().decode()
            debug(out)
            return variables, json.loads(out)
        else:
            out = p.stdout.read().decode()
            warning(out)
            raise Exception('Execution failed.')
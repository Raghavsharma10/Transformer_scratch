def _get_outputs(self):
        """
        Return a dict of the terraform outputs.

        :return: dict of terraform outputs
        :rtype: dict
        """
        if self.tf_version >= (0, 7, 0):
            logger.debug('Running: terraform output')
            res = self._run_tf('output', cmd_args=['-json'])
            outs = json.loads(res.strip())
            res = {}
            for k in outs.keys():
                if isinstance(outs[k], type({})):
                    res[k] = outs[k]['value']
                else:
                    res[k] = outs[k]
            logger.debug('Terraform outputs: %s', res)
            return res
        logger.debug('Running: terraform output')
        res = self._run_tf('output')
        outs = {}
        for line in res.split("\n"):
            line = line.strip()
            if line == '':
                continue
            parts = line.split(' = ', 1)
            outs[parts[0]] = parts[1]
        logger.debug('Terraform outputs: %s', outs)
        return outs
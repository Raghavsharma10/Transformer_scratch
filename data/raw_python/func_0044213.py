def _run_tf(self, cmd, cmd_args=[], stream=False):
        """
        Run a single terraform command via :py:func:`~.utils.run_cmd`;
        raise exception on non-zero exit status.

        :param cmd: terraform command to run
        :type cmd: str
        :param cmd_args: arguments to command
        :type cmd_args: :std:term:`list`
        :return: command output
        :rtype: str
        :raises: Exception on non-zero exit
        """
        args = [self.tf_path, cmd] + cmd_args
        arg_str = ' '.join(args)
        logger.info('Running terraform command: %s', arg_str)
        out, retcode = run_cmd(arg_str, stream=stream)
        if retcode != 0:
            logger.critical('Terraform command (%s) failed with exit code '
                            '%d:\n%s', arg_str, retcode, out)
            raise Exception('terraform %s failed' % cmd)
        return out
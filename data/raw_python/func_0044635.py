def _get_source(self):
        """
        Get the lambda function source template. Strip the leading docstring.
        Note that it's a real module in this project so we can test it.

        :return: function source code, with leading docstring stripped.
        :rtype: str
        """
        logger.debug('Getting module source for webhook2lambda2sqs.lambda_func')
        orig = getsourcelines(lambda_func)
        src = ''
        in_docstr = False
        have_docstr = False
        for line in orig[0]:
            if line.strip() == '"""' and not in_docstr and not have_docstr:
                in_docstr = True
                continue
            if line.strip() == '"""' and in_docstr:
                in_docstr = False
                have_docstr = True
                continue
            if not in_docstr:
                src += line
        return src
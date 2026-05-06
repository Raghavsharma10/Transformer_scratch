def generate(self, func_src):
        """
        Generate TF config and write to ./webhook2lambda2sqs.tf.json;
        write the lambda function to ./webhook2lambda2sqs_func.py

        :param func_src: lambda function source
        :type func_src: str
        """
        # write function source for reference
        logger.warning('Writing lambda function source to: '
                       './webhook2lambda2sqs_func.py')
        with open('./webhook2lambda2sqs_func.py', 'w') as fh:
            fh.write(func_src)
        logger.debug('lambda function written')
        # write upload zip
        logger.warning('Writing lambda function source zip file to: '
                       './webhook2lambda2sqs_func.zip')
        self._write_zip(func_src, './webhook2lambda2sqs_func.zip')
        logger.debug('lambda zip written')
        # write terraform
        logger.warning('Writing terraform configuration JSON to: '
                       './webhook2lambda2sqs.tf.json')
        with open('./webhook2lambda2sqs.tf.json', 'w') as fh:
            fh.write(self._get_config(func_src))
        logger.debug('terraform configuration written')
        logger.warning('Completed writing lambda function and TF config.')
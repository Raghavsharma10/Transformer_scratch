def inform_of_short_bin_name(cls, binary):
        """Historically, we had "devassistant" binary, but we chose to go with
        shorter "da". We still allow "devassistant", but we recommend using "da".
        """
        binary = os.path.splitext(os.path.basename(binary))[0]
        if binary != 'da':
            msg = '"da" is the preffered way of running "{binary}".'.format(binary=binary)
            logger.logger.info('*' * len(msg))
            logger.logger.info(msg)
            logger.logger.info('*' * len(msg))
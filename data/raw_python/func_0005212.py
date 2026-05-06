def main():
    """Sample usage for this python module

    This main method simply illustrates sample usage for this python
    module.

    :return: None
    """
    log = logging.getLogger(Logify.get_name() + '.logify.main')
    log.info('logger name is: %s', Logify.get_name())
    log.debug('This is DEBUG')
    log.info('This is INFO')
    log.warning('This is a WARNING')
    log.error('This is an ERROR')
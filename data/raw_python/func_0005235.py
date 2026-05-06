def main():
    """Sample usage for this python module

    This main method simply illustrates sample usage for this python
    module.

    :return: None
    """
    log = logging.getLogger(mod_logger + '.main')
    log.debug('This is DEBUG!')
    log.info('This is INFO!')
    log.warning('This is WARNING!')
    log.error('This is ERROR!')
    log.info('Running s3util.main...')
    my_bucket = 'cons3rt-deploying-cons3rt'
    my_regex = 'sourcebuilder.*apache-maven-.*3.3.3.*'
    try:
        s3util = S3Util(my_bucket)
    except S3UtilError as e:
        log.error('There was a problem creating S3Util:\n%s', e)
    else:
        log.info('Created S3Util successfully')
        key = s3util.find_key(my_regex)
        test = None
        if key is not None:
            test = s3util.download_file(key, '/Users/yennaco/Downloads')
        if test is not None:
            upload = s3util.upload_file(test, 'media-files-offline-assets/test')
            log.info('Upload result: %s', upload)
    log.info('End of main!')
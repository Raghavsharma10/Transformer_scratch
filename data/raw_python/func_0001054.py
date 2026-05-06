def _register_mecab_loc(location):
    ''' Set MeCab binary location '''
    global MECAB_LOC
    if not os.path.isfile(location):
        logging.getLogger(__name__).warning("Provided mecab binary location does not exist {}".format(location))
    logging.getLogger(__name__).info("Mecab binary is switched to: {}".format(location))
    MECAB_LOC = location
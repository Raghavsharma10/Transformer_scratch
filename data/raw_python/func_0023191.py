def num(string):
    """convert a string to float"""
    if not isinstance(string, type('')):
        raise ValueError(type(''))
    try:
        string = re.sub('[^a-zA-Z0-9\.\-]', '', string)
        number = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", string)
        return float(number[0])
    except Exception as e:
        logger = logging.getLogger('tradingAPI.utils.num')
        logger.debug("number not found in %s" % string)
        logger.debug(e)
        return None
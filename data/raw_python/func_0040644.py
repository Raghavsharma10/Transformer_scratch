def getAllRegexp():
    ''' 
        Method that recovers ALL the list of <RegexpObject> classes to be processed....

        :return:    Returns a list [] of <RegexpObject> classes.
    '''
    logger = logging.getLogger("entify")

    logger.debug("Recovering all the available <RegexpObject> classes.")
    listAll = []
    # For demo only
    #listAll.append(Demo())
    listAll.append(BitcoinAddress())
    listAll.append(DNI())
    listAll.append(DogecoinAddress())         
    listAll.append(Email())
    listAll.append(IPv4())
    listAll.append(LitecoinAddress())
    listAll.append(MD5())
    listAll.append(NamecoinAddress())
    listAll.append(PeercoinAddress())
    listAll.append(SHA1())
    listAll.append(SHA256())
    listAll.append(URL())
    # Add any additional import here
    #listAll.append(AnyNewRegexp)
    # <ADD_NEW_REGEXP_TO_THE_LIST>
    # Please, notify the authors if you have written a new regexp.

    logger.debug("Returning a list of " + str(len(listAll)) + " <RegexpObject> classes.")
    return listAll
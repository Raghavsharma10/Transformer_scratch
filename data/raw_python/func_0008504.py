def xml_encode(string):
    """ Returns the string with XML-safe special characters.
    """
    string = string.replace("&", "&amp;")
    string = string.replace("<", "&lt;")
    string = string.replace(">", "&gt;")
    string = string.replace("\"","&quot;")
    string = string.replace(SLASH, "/")
    return string
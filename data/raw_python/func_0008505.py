def xml_decode(string):
    """ Returns the string with special characters decoded.
    """
    string = string.replace("&amp;", "&")
    string = string.replace("&lt;",  "<")
    string = string.replace("&gt;",  ">")
    string = string.replace("&quot;","\"")
    string = string.replace("/", SLASH)
    return string
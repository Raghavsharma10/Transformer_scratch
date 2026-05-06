def listEverything(matching=False):
    """Prints every page in the project to the console.

    Args:
        matching (str, optional): if given, only return names with this string in it

    """
    pages=pageNames()
    if matching:
        pages=[x for x in pages if matching in x]
    for i,page in enumerate(pages):
        pages[i]="%s%s (%s)"%(pageFolder(page),page,getPageType(page))
    print("\n".join(sorted(pages)))
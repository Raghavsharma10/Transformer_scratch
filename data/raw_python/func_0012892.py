def folder2ver(folder):
    """get the version number from the E+ install folder"""
    ver = folder.split('EnergyPlus')[-1]
    ver = ver[1:]
    splitapp = ver.split('-')
    ver = '.'.join(splitapp)
    return ver
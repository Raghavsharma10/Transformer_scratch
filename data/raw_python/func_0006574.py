def checkInstalledPip(package, speak=True, speakSimilar=True):
    """checks if a given package is installed on pip"""
    packages = sorted([i.key for i in pip.get_installed_distributions()])
    installed = package in packages
    similar = None

    if not installed:
        similar = [pkg for pkg in packages if package in pkg]

    if speak:
        speakInstalledPackages(package, "pip", installed, similar, speakSimilar)

    return (installed, similar)
def checkInstalledBrew(package, similar=True, speak=True, speakSimilar=True):
    """checks if a given package is installed on homebrew"""
    packages = subprocess.check_output(['brew', 'list']).split()
    installed = package in packages
    similar = []

    if not installed:
        similar = [pkg for pkg in packages if package in pkg]
    if speak:
        speakInstalledPackages(package, "homebrew", installed, similar, speakSimilar)

    return (installed, similar)
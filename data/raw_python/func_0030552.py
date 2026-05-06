def pom_contains_modules():
    """
    Reads pom.xml in current working directory and checks, if there is non-empty modules tag.
    """
    pom_file = None
    try:
        pom_file = open("pom.xml")
        pom = pom_file.read()
    finally:
        if pom_file:
            pom_file.close()

    artifact = MavenArtifact(pom=pom)
    if artifact.modules:
        return True
    else:
        return False
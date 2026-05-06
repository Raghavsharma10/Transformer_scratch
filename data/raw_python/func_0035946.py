def genExampleStar(binaryLetter='', heirarchy=True):
    """ generates example star, if binaryLetter is true creates a parent binary object, if heirarchy is true will create a
    system and link everything up
    """

    starPar = StarParameters()
    starPar.addParam('age', '7.6')
    starPar.addParam('magB', '9.8')
    starPar.addParam('magH', '7.4')
    starPar.addParam('magI', '7.6')
    starPar.addParam('magJ', '7.5')
    starPar.addParam('magK', '7.3')
    starPar.addParam('magV', '9.0')
    starPar.addParam('mass', '0.98')
    starPar.addParam('metallicity', '0.43')
    starPar.addParam('name', 'Example Star {0}{1}'.format(ac._ExampleSystemCount, binaryLetter))
    starPar.addParam('name', 'HD {0}{1}'.format(ac._ExampleSystemCount, binaryLetter))
    starPar.addParam('radius', '0.95')
    starPar.addParam('spectraltype', 'G5')
    starPar.addParam('temperature', '5370')

    exampleStar = Star(starPar.params)
    exampleStar.flags.addFlag('Fake')

    if heirarchy:
        if binaryLetter:
            exampleBinary = genExampleBinary()
            exampleBinary._addChild(exampleStar)
            exampleStar.parent = exampleBinary
        else:
            exampleSystem = genExampleSystem()
            exampleSystem._addChild(exampleStar)
            exampleStar.parent = exampleSystem

    return exampleStar
def genExamplePlanet(binaryLetter=''):
    """ Creates a fake planet with some defaults
    :param `binaryLetter`: host star is part of a binary with letter binaryletter
    :return:
    """

    planetPar = PlanetParameters()
    planetPar.addParam('discoverymethod', 'transit')
    planetPar.addParam('discoveryyear', '2001')
    planetPar.addParam('eccentricity', '0.09')
    planetPar.addParam('inclination', '89.2')
    planetPar.addParam('lastupdate', '12/12/08')
    planetPar.addParam('mass', '3.9')
    planetPar.addParam('name', 'Example Star {0}{1} b'.format(ac._ExampleSystemCount, binaryLetter))
    planetPar.addParam('period', '111.2')
    planetPar.addParam('radius', '0.92')
    planetPar.addParam('semimajoraxis', '0.449')
    planetPar.addParam('temperature', '339.6')
    planetPar.addParam('transittime', '2454876.344')
    planetPar.addParam('separation', '330', {'unit': 'AU'})

    examplePlanet = Planet(planetPar.params)
    examplePlanet.flags.addFlag('Fake')

    exampleStar = genExampleStar(binaryLetter=binaryLetter)
    exampleStar._addChild(examplePlanet)
    examplePlanet.parent = exampleStar

    return examplePlanet
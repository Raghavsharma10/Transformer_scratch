def generateVectors():
    """Convert the known ra/decs of the channel corners
    into unit vectors. This code creates the conents of the
    function loadOriginVectors() (below)
    """

    ra_deg = 290.66666667
    dec_deg = +44.5
    #rollAngle_deg = 33.0
    rollAngle_deg = +123.
    boresight = r.vecFromRaDec(ra_deg, dec_deg)

    #Read in prime mission coords and convert to vectors
    inFile = "../etc/fov.txt"
    inFile = np.loadtxt(inFile)
    radecs = np.zeros( (len(inFile), 3))
    rotate = radecs*0
    for i, row in enumerate(inFile):
        radecs[i, :] = r.vecFromRaDec(inFile[i, 4], inFile[i,5])
        rotate[i, :] = r.rotateAroundVector(radecs[i], boresight,\
            -rollAngle_deg)




    #Slew to ra/dec of zero
    Ra = r.rightAscensionRotationMatrix(-ra_deg)
    Rd = r.declinationRotationMatrix(-dec_deg)
    R = np.dot(Rd, Ra)
    origin = rotate*0
    for i, row in enumerate(rotate):
        origin[i] = np.dot(R, rotate[i])

    mp.plot(origin[:,0], origin[:,1])

    #Print out the results
    #import pdb; pdb.set_trace()
    print("[")
    for i in range(len(inFile)):
        ch = channelFromModOut(inFile[i,0], inFile[i,1])
        print("[%3i., %3i., %3i., %13.7f, %13.7f, %13.7f], \\" %( \
            inFile[i, 0], inFile[i, 1], ch, \
            origin[i, 0], origin[i, 1], origin[i,2]))
    print("]")
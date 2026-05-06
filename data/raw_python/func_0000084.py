def runcommand(cosmology='WMAP5'):
    """ Example interface commands """

    # Return the WMAP5 cosmology concentration predicted for
    # z=0 range of masses
    Mi = [1e8, 1e9, 1e10]
    zi = 0
    print("Concentrations for haloes of mass %s at z=%s" % (Mi, zi))
    output = commah.run(cosmology=cosmology, zi=zi, Mi=Mi)

    print(output['c'].flatten())

    # Return the WMAP5 cosmology concentration predicted for
    # z=0 range of masses AND cosmological parameters
    Mi = [1e8, 1e9, 1e10]
    zi = 0
    print("Concentrations for haloes of mass %s at z=%s" % (Mi, zi))
    output, cosmo = commah.run(cosmology=cosmology, zi=zi, Mi=Mi,
                               retcosmo=True)

    print(output['c'].flatten())
    print(cosmo)

    # Return the WMAP5 cosmology concentration predicted for MW
    # mass (2e12 Msol) across redshift
    Mi = 2e12
    z = [0, 0.5, 1, 1.5, 2, 2.5]
    output = commah.run(cosmology=cosmology, zi=0, Mi=Mi, z=z)
    for zval in z:
        print("M(z=0)=%s has c(z=%s)=%s"
              % (Mi, zval, output[output['z'] == zval]['c'].flatten()))

    # Return the WMAP5 cosmology concentration predicted for MW
    # mass (2e12 Msol) across redshift
    Mi = 2e12
    zi = [0, 0.5, 1, 1.5, 2, 2.5]
    output = commah.run(cosmology=cosmology, zi=zi, Mi=Mi)
    for zval in zi:
        print("M(z=%s)=%s has concentration %s"
              % (zval, Mi, output[(output['zi'] == zval) &
                                  (output['z'] == zval)]['c'].flatten()))

    # Return the WMAP5 cosmology concentration and
    # rarity of high-z cluster
    Mi = 2e14
    zi = 6
    output = commah.run(cosmology=cosmology, zi=zi, Mi=Mi)
    print("Concentrations for haloes of mass %s at z=%s" % (Mi, zi))
    print(output['c'].flatten())
    print("Mass variance sigma of haloes of mass %s at z=%s" % (Mi, zi))
    print(output['sig'].flatten())
    print("Fluctuation for haloes of mass %s at z=%s" % (Mi, zi))
    print(output['nu'].flatten())

    # Return the WMAP5 cosmology accretion rate prediction
    # for haloes at range of redshift and mass
    Mi = [1e8, 1e9, 1e10]
    zi = [0]
    z = [0, 0.5, 1, 1.5, 2, 2.5]
    output = commah.run(cosmology=cosmology, zi=zi, Mi=Mi, z=z)
    for Mval in Mi:
        print("dM/dt for halo of mass %s at z=%s across redshift %s is: "
              % (Mval, zi, z))
        print(output[output['Mi'] == Mval]['dMdt'].flatten())

    # Return the WMAP5 cosmology Halo Mass History for haloes with M(z=0) = 1e8
    M = [1e8]
    z = [0, 0.5, 1, 1.5, 2, 2.5]
    print("Halo Mass History for z=0 mass of %s across z=%s" % (M, z))
    output = commah.run(cosmology=cosmology, zi=0, Mi=M, z=z)
    print(output['Mz'].flatten())

    # Return the WMAP5 cosmology formation redshifts for haloes at
    # range of redshift and mass
    M = [1e8, 1e9, 1e10]
    z = [0]
    print("Formation Redshifts for haloes of mass %s at z=%s" % (M, z))
    output = commah.run(cosmology=cosmology, zi=0, Mi=M, z=z)
    for Mval in M:
        print(output[output['Mi'] == Mval]['zf'].flatten())

    return("Done")
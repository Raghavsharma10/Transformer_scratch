def zSetDefaultMeritFunctionSEQ(self, ofType=0, ofData=0, ofRef=0, pupilInteg=0, rings=0,
                                    arms=0, obscuration=0, grid=0, delVignetted=False, useGlass=False, 
                                    glassMin=0, glassMax=1000, glassEdge=0, useAir=False, airMin=0, 
                                    airMax=1000, airEdge=0, axialSymm=True, ignoreLatCol=False, 
                                    addFavOper=False, startAt=1, relativeXWgt=1.0, overallWgt=1.0, 
                                    configNum=0):
        """Sets the default merit function for Sequential Merit Function Editor

        Parameters
        ----------
        ofType : integer
            optimization function type (0=RMS, ...)
        ofData : integer 
            optimization function data (0=Wavefront, 1=Spot Radius, ...)
        ofRef : integer
            optimization function reference (0=Centroid, ...)
        pupilInteg : integer
            pupil integration method (0=Gaussian Quadrature, 1=Rectangular Array)
        rings : integer
            rings (0=1, 1=2, 2=3, 3=4, ...)
        arms : integer 
            arms (0=6, 1=8, 2=10, 3=12)
        obscuration : real
            obscuration
        delVignetted : boolean 
            delete vignetted ?
        useGlass : boolean 
            whether to use Glass settings for thickness boundary
        glassMin : real
            glass mininum thickness 
        glassMax : real 
            glass maximum thickness
        glassEdge : real
            glass edge thickness
        useAir : boolean
            whether to use Air settings for thickness boundary
        airMin : real
            air minimum thickness      
        airMax : real 
            air maximum thickness
        airEdge : real 
            air edge thickness
        axialSymm : boolean 
            assume axial symmetry 
        ignoreLatCol : boolean
            ignore latent color
        addFavOper : boolean
            add favorite color
        configNum : integer
            configuration number (0=All)
        startAt : integer 
            start at
        relativeXWgt : real 
            relative X weight
        overallWgt : real
            overall weight
        """
        mfe = self.pMFE
        wizard = mfe.pSEQOptimizationWizard
        wizard.pType = ofType
        wizard.pData = ofData
        wizard.pReference = ofRef
        wizard.pPupilIntegrationMethod = pupilInteg 
        wizard.pRing = rings
        wizard.pArm = arms
        wizard.pObscuration = obscuration
        wizard.pGrid = grid
        wizard.pIsDeleteVignetteUsed =  delVignetted
        wizard.pIsGlassUsed = useGlass 
        wizard.pGlassMin = glassMin
        wizard.pGlassMax = glassMax
        wizard.pGlassEdge = glassEdge
        wizard.pIsAirUsed = useAir
        wizard.pAirMin = airMin
        wizard.pAirMax = airMax 
        wizard.pAirEdge = airEdge 
        wizard.pIsAssumeAxialSymmetryUsed = axialSymm
        wizard.pIsIgnoreLateralColorUsed = ignoreLatCol
        wizard.pConfiguration = configNum 
        wizard.pIsAddFavoriteOperandsUsed = addFavOper
        wizard.pStartAt = startAt
        wizard.pRelativeXWeight = relativeXWgt
        wizard.pOverallWeight = overallWgt
        wizard.CommonSettings.OK()
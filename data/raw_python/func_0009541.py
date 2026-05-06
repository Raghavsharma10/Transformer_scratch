def EL_Si_module():
    '''
    returns angular dependent EL emissivity of a PV module
    
    calculated of nanmedian(persp-corrected EL module/reference module)
    
    published in K. Bedrich: Quantitative Electroluminescence Measurement on PV devices
                 PhD Thesis, 2017
    '''
    arr = np.array([
                    [2.5, 1.00281 ],
                    [7.5, 1.00238 ],
                    [12.5, 1.00174],
                    [17.5, 1.00204 ],
                    [22.5, 1.00054 ],
                    [27.5, 0.998255],
                    [32.5, 0.995351],
                    [37.5, 0.991246],
                    [42.5, 0.985304],
                    [47.5, 0.975338],
                    [52.5, 0.960455],
                    [57.5, 0.937544],
                    [62.5, 0.900607],
                    [67.5, 0.844636],
                    [72.5, 0.735028],
                    [77.5, 0.57492 ],
                    [82.5, 0.263214],
                    [87.5, 0.123062]
                    ])

    angles = arr[:,0]
    vals = arr[:,1]

    vals[vals>1]=1
    return angles, vals
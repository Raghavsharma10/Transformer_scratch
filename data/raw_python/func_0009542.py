def TG_glass():
    '''
    reflected temperature for 250DEG Glass
    published in IEC 62446-3 TS: Photovoltaic (PV) systems 
    - Requirements for testing, documentation and maintenance 
    - Part 3: Outdoor infrared thermography of photovoltaic modules 
      and plants p Page 12
    '''
    vals = np.array([(80,0.88),
                     (75,0.88),
                     (70,0.88),
                     (65,0.88),
                     (60,0.88),
                     (55,0.88),
                     (50,0.87),
                     (45,0.86),
                     (40,0.85),
                     (35,0.83),
                     (30,0.80),
                     (25,0.76),
                     (20,0.7),
                     (15,0.60),
                     (10,0.44)])
    #invert angle reference:
    vals[:,0]=90-vals[:,0]
    #make emissivity relative:
    vals[:,1]/=vals[0,1]
    return vals[:,0], vals[:,1]
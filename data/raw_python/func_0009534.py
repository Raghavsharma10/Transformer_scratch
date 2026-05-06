def interpolateCircular2dStructuredIDW(grid, mask, kernel=15, power=2, 
                               fr=1, fphi=1, cx=0, cy=0):
    '''
    same as interpolate2dStructuredIDW
    but calculation distance to neighbour using polar coordinates
    fr, fphi --> weight factors for radian and radius differences
    cx,cy -> polar center of the array e.g. middle->(sx//2+1,sy//2+1) 
    '''
    gx = grid.shape[0]
    gy = grid.shape[0]

    #FOR EVERY PIXEL
    for i in range(gx):
        for j in range(gy):
            
            if mask[i,j]:
                
                xmn = i-kernel
                if xmn < 0:
                    xmn = 0
                xmx = i+kernel
                if xmx > gx:
                    xmx = gx
                    
                ymn = j-kernel
                if ymn < 0:
                    ymn = 0
                ymx = j+kernel
                if ymx > gx:
                    ymx = gy
                    
                sumWi = 0.0
                value = 0.0 

                #radius and radian to polar center:
                R = ((i-cx)**2+(j-cy)**2)**0.5
                PHI = atan2(j-cy, i-cx)
                
                #FOR EVERY NEIGHBOUR IN KERNEL              
                for xi in range(xmn,xmx):
                    for yi in range(ymn,ymx):
                        if  (xi != i or yi != j) and not mask[xi,yi]:
                            nR = ((xi-cx)**2+(yi-cy)**2)**0.5
                            dr =  R - nR
                            #average radius between both p:
                            midR = 0.5*(R+nR)
                            #radian of neighbour p:
                            nphi = atan2(yi-cy, xi-cx)
                            #relative angle between both points:
                            dphi = min((2*np.pi) - abs(PHI - nphi), 
                                       abs(PHI - nphi))    
                            dphi*=midR       
                            
                            dist = ((fr*dr)**2+(fphi*dphi)**2)**2

                            wi = 1 / dist**(0.5*power)
                            sumWi += wi
                            value += wi * grid[xi,yi]  
                if sumWi:
                    grid[i,j] = value / sumWi               

    return grid
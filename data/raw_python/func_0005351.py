def generate_synthObs(self, bases_wave, bases_flux, basesCoeff, Av_star, z_star, sigma_star, resample_range = None, resample_int = 1):
        
        '''basesWave: Bases wavelength must be at rest'''
        nbases              = basesCoeff.shape[0]        
        bases_wave_resam    = arange(int(resample_range[0]), int(resample_range[-1]), resample_int, dtype=float)
        npix_resample       = len(bases_wave_resam)
        
        #Resampling the range
        bases_flux_resam = empty((nbases, npix_resample))
        for i in range(nbases):
#             print bases_wave[i][0], bases_wave[i][-1]
#             print bases_wave_resam[0], bases_wave_resam[-1]
            bases_flux_resam[i,:] = interp1d(bases_wave[i], bases_flux[i], bounds_error=True)(bases_wave_resam)            
        
        #Display physical parameters
        synth_wave                  = bases_wave_resam * (1 + z_star)
        
        Av_vector                   = Av_star * ones(nbases)
        Xx_redd                     = CCM89_Bal07(3.4, bases_wave_resam)   
        r_sigma                     = sigma_star/(synth_wave[1] - synth_wave[0])

        #Defining empty kernel
        box                         = int(3 * r_sigma) if int(3 * r_sigma) < 3 else 3
        kernel_len                  = 2 * box + 1
        kernel                      = zeros((1, kernel_len)) 
        kernel_range                = arange(0, 2 * box + 1)
        
        #Generating the kernel with sigma (the norm factor is the sum of the gaussian)
        kernel[0,:]                 = exp(-0.5 * ((square(kernel_range-box)/r_sigma)))
        norm                        = np_sum(kernel[0,:])        
        kernel                      = kernel / norm

        #Convove bases with respect to kernel for dispersion velocity calculation
        bases_grid_convolve         = convolve2d(bases_flux_resam, kernel, mode='same', boundary='symm')  
        
        #Interpolate bases to wavelength range
        interBases_matrix           = (interp1d(bases_wave_resam, bases_grid_convolve, axis=1, bounds_error=True)(bases_wave_resam)).T       

        #Generate final flux model including dust        
        dust_attenuation            = power(10, -0.4 * outer(Xx_redd, Av_vector))
        bases_grid_model            = interBases_matrix * dust_attenuation
                    
        #Generate combined flux
        synth_flux                  = np_sum(basesCoeff.T * bases_grid_model, axis=1)
                                      
        return synth_wave, synth_flux
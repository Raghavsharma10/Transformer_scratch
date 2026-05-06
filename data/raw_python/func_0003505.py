def albedo(self, model='smith'):
        """Finds broad-band surface reflectance (albedo)
        
        Smith (2010),  “The heat budget of the earth’s surface deduced from space”
        LT5 toa reflectance bands 1, 3, 4, 5, 7
        # normalized i.e. 0.356 + 0.130 + 0.373 + 0.085 + 0.07 = 1.014
        
        Should have option for Liang, 2000; 
        
        Tasumi (2008), "At-Surface Reflectance and Albedo from Satellite for
                        Operational Calculation of Land Surface Energy Balance"
                        
        
        :return albedo array of floats
        """
        if model == 'smith':
            blue, red, nir, swir1, swir2 = (self.reflectance(1), self.reflectance(3), self.reflectance(4),
                                            self.reflectance(5), self.reflectance(7))
            alb = (0.356 * blue + 0.130 * red + 0.373 * nir + 0.085 * swir1 + 0.072 * swir2 - 0.0018) / 1.014
        elif model == 'tasumi':
            pass
        # add tasumi algorithm TODO
        return alb
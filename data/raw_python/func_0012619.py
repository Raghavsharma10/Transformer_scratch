def sample(self):
        """
        Draws a trajectory length, first coordinates, lengths, angles and 
        length-angle-difference pairs according to the empirical distribution. 
        Each call creates one complete trajectory.
        """
        lenghts = []
        angles = []
        coordinates = []
        fix = []
        sample_size = int(round(self.trajLen_borders[self.drawFrom('self.trajLen_cumsum', self.getrand('self.trajLen_cumsum'))]))

        coordinates.append([0, 0])
        fix.append(1)
        
        while len(coordinates) < sample_size:
            if len(lenghts) == 0 and len(angles) == 0:          
                angle, length = self._draw(self)
            else:
                angle, length = self._draw(prev_angle = angles[-1], 
                                            prev_length = lenghts[-1])  
                        
            x, y = self._calc_xy(coordinates[-1], angle, length) 
            
            coordinates.append([x, y])
            lenghts.append(length) 
            angles.append(angle)
            fix.append(fix[-1]+1)
        return coordinates
def _draw(self, prev_angle = None, prev_length = None):
        """
        Draws a new length- and angle-difference pair and calculates
        length and angle absolutes matching the last saccade drawn.

        Parameters:
            prev_angle : float, optional
                The last angle that was drawn in the current trajectory
            prev_length : float, optional
                The last length that was drawn in the current trajectory
            
            Note: Either both prev_angle and prev_length have to be given 
            or none; if only one parameter is given, it will be neglected.
        """
        
        if (prev_angle is None) or (prev_length is None):
            (length, angle)= np.unravel_index(self.drawFrom('self.firstLenAng_cumsum', self.getrand('self.firstLenAng_cumsum')),
                                                self.firstLenAng_shape)
            angle = angle-((self.firstLenAng_shape[1]-1)/2) 
            angle += 0.5
            length += 0.5
            length *= self.fm.pixels_per_degree
        else:
            ind = int(floor(prev_length/self.fm.pixels_per_degree))
            while ind >= len(self.probability_cumsum):
                ind -= 1

            while not(self.probability_cumsum[ind]).any():
                ind -= 1
                
            J, I = np.unravel_index(self.drawFrom('self.probability_cumsum '+repr(ind),self.getrand('self.probability_cumsum '+repr(ind))), 
                                    self.full_H1[ind].shape)
            angle = reshift((I-self.full_H1[ind].shape[1]/2) + prev_angle)
            angle += 0.5
            length = J+0.5
            length *= self.fm.pixels_per_degree
        return angle, length
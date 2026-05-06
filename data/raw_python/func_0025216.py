def set_maximum(self, q_data, marked, center, bin_lower, foothills):
        """
        Grow a region at a certain bin level and check if the region has reached the maximum size.

        Args:
            q_data: Quantized data array
            marked: Array marking points that are objects
            center: Coordinates of the center pixel of the region being grown
            bin_lower: Intensity level of lower bin being evaluated
            foothills: List of points that are associated with a center but fall outside the the size or
                intensity criteria
        Returns:
            True if the object is finished growing and False if the object should be grown again at the next
            threshold level.
        """
        as_bin = [] # pixels to be included in peak
        as_glob = []   # pixels to be globbed up as part of foothills
        marked_so_far = []  # pixels that have already been marked
        will_be_considered_again = False
        as_bin.append(center)
        center_data = q_data[center]
        while len(as_bin) > 0:
            p = as_bin.pop(-1) # remove and return last pixel in as_bin
            if marked[p] != self.UNMARKED: # already processed
                continue
            marked[p] = q_data[center]
            marked_so_far.append(p)

            # check neighbors
            for index,val in np.ndenumerate(marked[p[0] - 1:p[0] + 2, p[1] - 1:p[1] + 2]):
                # is neighbor part of peak or part of mountain?
                if val == self.UNMARKED:
                    pixel = (index[0] - 1 + p[0],index[1] - 1 + p[1])
                    p_data = q_data[pixel]
                    if (not will_be_considered_again) and (p_data >= 0) and (p_data < center_data):
                        will_be_considered_again = True
                    if p_data >= bin_lower and (np.abs(center_data - p_data) <= self.delta):
                        as_bin.append(pixel)
                    # Do not check that this is the closest: this way, a narrow channel of globbed pixels form
                    elif p_data >= 0:
                        as_glob.append(pixel)
        if bin_lower == 0:
            will_be_considered_again = False
        big_enough = len(marked_so_far) >= self.max_size
        if big_enough:
            # remove lower values within region of influence
            foothills.append((center, as_glob))
        elif will_be_considered_again: # remove the check if you want to ignore regions smaller than max_size
            for m in marked_so_far:
                marked[m] = self.UNMARKED
            del as_bin[:]
            del as_glob[:]
            del marked_so_far[:]
        return big_enough or (not will_be_considered_again)
def remove_foothills(self, q_data, marked, bin_num, bin_lower, centers, foothills):
        """
        Mark points determined to be foothills as globbed, so that they are not included in
        future searches. Also searches neighboring points to foothill points to determine
        if they should also be considered foothills.

        Args:
            q_data: Quantized data
            marked: Marked
            bin_num: Current bin being searched
            bin_lower: Next bin being searched
            centers: dictionary of local maxima considered to be object centers
            foothills: List of foothill points being removed.
        """
        hills = []
        for foot in foothills:
            center = foot[0]
            hills[:] = foot[1][:]
            # remove all foothills
            while len(hills) > 0:
                # mark this point
                pt = hills.pop(-1)
                marked[pt] = self.GLOBBED
                for s_index, val in np.ndenumerate(marked[pt[0]-1:pt[0]+2,pt[1]-1:pt[1]+2]):
                    index = (s_index[0] - 1 + pt[0], s_index[1] - 1 + pt[1])
                    # is neighbor part of peak or part of mountain?
                    if val == self.UNMARKED:
                        # will let in even minor peaks
                        if (q_data[index] >= 0) and \
                                (q_data[index] < bin_lower) and \
                                ((q_data[index] <= q_data[pt]) or self.is_closest(index, center, centers, bin_num)):
                            hills.append(index)
        del foothills[:]
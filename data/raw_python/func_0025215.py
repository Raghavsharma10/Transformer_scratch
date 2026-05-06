def find_local_maxima(self, input_grid):
        """
        Finds the local maxima in the inputGrid and perform region growing to identify objects.

        Args:
            input_grid: Raw input data.

        Returns:
            array with labeled objects.
        """
        pixels, q_data = self.quantize(input_grid)
        centers = OrderedDict()
        for p in pixels.keys():
            centers[p] = []
        marked = np.ones(q_data.shape, dtype=int) * self.UNMARKED
        MIN_INFL = int(np.round(1 + 0.5 * np.sqrt(self.max_size)))
        MAX_INFL = 2 * MIN_INFL
        marked_so_far = []
        # Find the maxima. These are high-values with enough clearance
        # around them.
        # Work from high to low bins. The pixels in the highest bin mark their
        # neighborhoods first. If you did it from low to high the lowest maxima
        # would mark their neighborhoods first and interfere with the identification of higher maxima.
        for b in sorted(pixels.keys(),reverse=True):
            # Square starts large with high intensity bins and gets smaller with low intensity bins.
            infl_dist = MIN_INFL + int(np.round(float(b) / self.max_bin * (MAX_INFL - MIN_INFL)))
            for p in pixels[b]:
                if marked[p] == self.UNMARKED:
                    ok = False
                    del marked_so_far[:]
                    # Temporarily mark unmarked points in square around point (keep track of them in list marked_so_far).
                    # If none of the points in square were marked already from a higher intensity center, 
                    # this counts as a new center and ok=True and points will remain marked.
                    # Otherwise ok=False and marked points that were previously unmarked will be unmarked.
                    for (i, j), v in np.ndenumerate(marked[p[0] - infl_dist:p[0] + infl_dist + 1,
                                                    p[1] - infl_dist:p[1]+ infl_dist + 1]):
                        if v == self.UNMARKED:
                            ok = True
                            marked[i - infl_dist + p[0],j - infl_dist + p[1]] = b
                           
                            marked_so_far.append((i - infl_dist + p[0],j - infl_dist + p[1]))
                        else:
                            # neighborhood already taken
                            ok = False
                            break
                    # ok if point and surrounding square were not marked already.
                    if ok:
                        # highest point in its neighborhood
                        centers[b].append(p)
                    else:
                        for m in marked_so_far:
                            marked[m] = self.UNMARKED
        # Erase marks and start over. You have a list of centers now.
        marked[:, :] = self.UNMARKED
        deferred_from_last = []
        deferred_to_next = []
        # delta (int): maximum number of increments the cluster is allowed to range over. Larger d results in clusters over larger scales.
        for delta in range(0, self.delta + 1):
            # Work from high to low bins.
            for b in sorted(centers.keys(), reverse=True):
                bin_lower = b - delta
                deferred_from_last[:] = deferred_to_next[:]
                del deferred_to_next[:]
                foothills = []
                n_centers = len(centers[b])
                tot_centers = n_centers + len(deferred_from_last)
                for i in range(tot_centers):
                    # done this way to minimize memory overhead of maintaining two lists
                    if i < n_centers:
                        center = centers[b][i]
                    else:
                        center = deferred_from_last[i - n_centers]
                    if bin_lower < 0:
                        bin_lower = 0
                    if marked[center] == self.UNMARKED:
                        captured = self.set_maximum(q_data, marked, center, bin_lower, foothills)
                        if not captured:
                            # decrement to lower value to see if it'll get big enough
                            deferred_to_next.append(center)
                        else:
                            pass
                # this is the last one for this bin
                self.remove_foothills(q_data, marked, b, bin_lower, centers, foothills)
            del deferred_from_last[:]
            del deferred_to_next[:]
        return marked
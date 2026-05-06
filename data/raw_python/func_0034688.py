def intersect(self, other):
        """ Computes the multiset intersection, between the current Multicolor and the supplied Multicolor

        :param other: another Multicolor object to compute a multiset intersection with
        :return:
        :raise TypeError: an intersection can be computed only between two Multicolor objects
        """
        if not isinstance(other, Multicolor):
            raise TypeError("Multicolor can be intersected only with another Multicolor object")
        intersection_colors_core = self.colors.intersection(other.colors)
        colors_count = {color: min(self.multicolors[color], other.multicolors[color]) for color in intersection_colors_core}
        return Multicolor(*(color for color in colors_count for _ in range(colors_count[color])))
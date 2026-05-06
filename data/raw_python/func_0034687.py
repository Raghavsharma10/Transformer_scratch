def split_colors(cls, multicolor, guidance=None, sorted_guidance=False,
                     account_for_color_multiplicity_in_guidance=True):
        """ Produces several new instances of :class:`Multicolor` object by splitting information about colors by using provided guidance iterable set-like object.

        Guidance is an iterable type of object where each entry has information about groups of colors that has to be separated for current :attr:`Multicolor.multicolors` chunk.
        If no Guidance is provided, single-color guidance of :attr:`Multicolor.multicolors` is created.
        Guidance object is first reversed sorted to iterate over it from larges color set to the smallest one, as small color sets might be subsets of bigger ones, and shall be utilized only if bigger sets didn't help in separating.

        During the first iteration over the guidance information all subsets of :attr:`Multicolor.multicolors` that equal to entries of guidance are recorded.
        During second iteration over remaining of the guidance information, if colors in :attr:`Multicolor.multicolors` form subsets of guidance entries, such instances are recorded.
        After this two iterations, the rest of :attr:`Multicolor.multicolors` is recorded as non-tackled and is recorded on its own.

        Multiplicity of all separated colors in respective chunks is preserved.

        Accounts for subclassing.

        :param multicolor: an instance information about colors in which is to be split
        :type multicolor: :class:`Multicolor`
        :param guidance: information how colors have to be split in current :class:`Multicolor` object
        :type guidance: iterable where each entry is iterable with colors entries
        :param sorted_guidance: a flag, that indicates is sorting of provided guidance is in order
        :return: a list of new :class:`Multicolor` object colors information in which complies with guidance information
        :rtype: ``list`` of :class:`Multicolor` objects
        """
        if guidance is None:
            ###############################################################################################
            #
            # if guidance is not specified, it will be derived from colors in the targeted multicolor
            # initially the multiplicity of colors remains as is
            #
            ###############################################################################################
            guidance = [Multicolor(*(color for _ in range(multicolor.multicolors[color]))) for color in multicolor.colors]
            ###############################################################################################
            #
            # since at this we have a single-colored (maybe with multiplicity greater than 1)
            # we don't need to sort anything, as there will be no overlapping multicolor in guidance
            #
            ###############################################################################################
            sorted_guidance = True
        ###############################################################################################
        #
        # a reference to the targeted multicolor.
        # such reference is created only for the future requirement to access information about original multicolor
        # Is done for the sake of code clarity and consistency.
        #
        ###############################################################################################
        splitting_multicolor = deepcopy(multicolor)
        if not account_for_color_multiplicity_in_guidance:
            ###############################################################################################
            #
            # we need to create a new guidance (even if original is perfect)
            # a new one shall preserve the order of the original, but all multicolors in it
            #   while keeping information about the actual colors itself, shall have multiplicity equal to 1
            #
            ###############################################################################################
            splitting_multicolor = Multicolor(*multicolor.colors)
            colors_guidance = [Multicolor(*tmp_multicolor.colors) for tmp_multicolor in guidance]
            ###############################################################################################
            #
            # since there might be different multicolors, with the same colors content
            # and they will be changed to same multicolors object, after colors multiplicity adjustment
            # we need, while preserving the order, leave only unique ones in (the first appearance)
            #
            ###############################################################################################
            unique = set()
            guidance = []
            for c_multicolor in colors_guidance:
                if c_multicolor.hashable_representation not in unique:
                    unique.add(c_multicolor.hashable_representation)
                    guidance.append(c_multicolor)
        if not sorted_guidance:
            ###############################################################################################
            #
            # if arguments in function call do not specify explicitly, that the guidance shall be used "as is"
            # it is sorted to put "bigger" multicolors in front, and smaller at the back
            # as bigger multicolor might contain several smaller multicolors from the guidance, but the correct splitting
            # always assumes that the smaller is the splitted result, the better it is
            # and such minimization can be obtained only if the biggest chunk of targeted multicolor are ripped off of it first
            #
            ###############################################################################################
            guidance = sorted({g_multicolor.hashable_representation for g_multicolor in guidance},
                              key=lambda g_multicolor: len(g_multicolor),
                              reverse=True)
            guidance = [Multicolor(*hashed) for hashed in guidance]
        first_run_result = []
        second_run_result = []
        for g_multicolor in guidance:
            ###############################################################################################
            #
            # first we determine which multicolors in guidance are fully present in the multicolor to split
            #   "<=" operator can be read as "is_multi_subset_of"
            # and retrieve as many copies of it from the multicolor, as we can
            # Example:
            #   multicolor has only one color "blue" with multiplicity "4"
            #   in guidance we have multicolor with color "blue" with multiplicity "2"
            #   we must retrieve it fully twice
            #
            ###############################################################################################
            ###############################################################################################
            #
            # empty guidance multicolors shall be ignored, as they have no impact on the splitting algorithm
            #
            ###############################################################################################
            if len(g_multicolor.colors) == 0:
                continue
            while g_multicolor <= splitting_multicolor:
                first_run_result.append(g_multicolor)
                splitting_multicolor -= g_multicolor
        for g_multicolor in guidance:
            ###############################################################################################
            #
            # secondly we determine which multicolors in guidance are partially present in the multicolor
            # NOTE that this is not possible for the case of tree consistent multicolor
            #   as every partially present
            #
            ###############################################################################################
            while len(g_multicolor.intersect(splitting_multicolor).multicolors) > 0:
                second_run_result.append(g_multicolor.intersect(splitting_multicolor))
                splitting_multicolor -= g_multicolor.intersect(splitting_multicolor)
        appendix = splitting_multicolor
        result = deepcopy(first_run_result) + deepcopy(second_run_result) + deepcopy([appendix] if len(appendix.multicolors) > 0 else [])
        if not account_for_color_multiplicity_in_guidance:
            ###############################################################################################
            #
            # if we didn't care for guidance multicolors colors multiplicity, we we splitting a specially created Multicolor
            # based only on the colors content.
            # After this is done, we need to restore the original multiplicity of each color in result multicolors to the
            # count they had in the targeted for splitting multicolor.
            # This is possible since in the case when we do not account for colors multiplicity in guidance, we have
            # splitting_color variable referencing not the supplied multicolor, and thus internal changes are not made to
            # supplied multicolor.
            #
            ###############################################################################################
            for r_multicolor in result:
                for color in r_multicolor.colors:
                    r_multicolor.multicolors[color] = multicolor.multicolors[color]
        return result
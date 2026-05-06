def page(self, number, *args, **kwargs):
        """Return a standard ``Page`` instance with custom, digg-specific
        page ranges attached.
        """

        page = super().page(number, *args, **kwargs)
        number = int(number) # we know this will work

        # easier access
        num_pages, body, tail, padding, margin = \
            self.num_pages, self.body, self.tail, self.padding, self.margin

        # put active page in middle of main range
        main_range = list(map(int, [
            math.floor(number-body/2.0)+1,  # +1 = shift odd body to right
            math.floor(number+body/2.0)]))
        # adjust bounds
        if main_range[0] < 1:
            main_range = list(map(abs(main_range[0]-1).__add__, main_range))
        if main_range[1] > num_pages:
            main_range = list(map((num_pages-main_range[1]).__add__, main_range))

        # Determine leading and trailing ranges; if possible and appropriate,
        # combine them with the main range, in which case the resulting main
        # block might end up considerable larger than requested. While we
        # can't guarantee the exact size in those cases, we can at least try
        # to come as close as possible: we can reduce the other boundary to
        # max padding, instead of using half the body size, which would
        # otherwise be the case. If the padding is large enough, this will
        # of course have no effect.
        # Example:
        #     total pages=100, page=4, body=5, (default padding=2)
        #     1 2 3 [4] 5 6 ... 99 100
        #     total pages=100, page=4, body=5, padding=1
        #     1 2 3 [4] 5 ... 99 100
        # If it were not for this adjustment, both cases would result in the
        # first output, regardless of the padding value.
        if main_range[0] <= tail+margin:
            leading = []
            main_range = [1, max(body, min(number+padding, main_range[1]))]
            main_range[0] = 1
        else:
            leading = list(range(1, tail+1))
        # basically same for trailing range, but not in ``left_align`` mode
        if self.align_left:
            trailing = []
        else:
            if main_range[1] >= num_pages-(tail+margin)+1:
                trailing = []
                if not leading:
                    # ... but handle the special case of neither leading nor
                    # trailing ranges; otherwise, we would now modify the
                    # main range low bound, which we just set in the previous
                    # section, again.
                    main_range = [1, num_pages]
                else:
                    main_range = [min(num_pages-body+1, max(number-padding, main_range[0])), num_pages]
            else:
                trailing = list(range(num_pages-tail+1, num_pages+1))

        # finally, normalize values that are out of bound; this basically
        # fixes all the things the above code screwed up in the simple case
        # of few enough pages where one range would suffice.
        main_range = [max(main_range[0], 1), min(main_range[1], num_pages)]

        # make the result of our calculations available as custom ranges
        # on the ``Page`` instance.
        page.main_range = list(range(main_range[0], main_range[1]+1))
        page.leading_range = leading
        page.trailing_range = trailing
        page.page_range = reduce(lambda x, y: x+((x and y) and [False])+y,
            [page.leading_range, page.main_range, page.trailing_range])

        page.__class__ = DiggPage
        return page
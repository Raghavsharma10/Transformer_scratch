def parseFilteringOptions(cls, args, filterRead=None, storeQueryIds=False):
        """
        Parse command line options (added in C{addSAMFilteringOptions}.

        @param args: The command line arguments, as returned by
            C{argparse.parse_args}.
        @param filterRead: A one-argument function that accepts a read
            and returns C{None} if the read should be omitted in filtering
            or else a C{Read} instance.
        @param storeQueryIds: If C{True}, query ids will be stored as the
            SAM/BAM file is read.
        @return: A C{SAMFilter} instance.
        """
        referenceIds = (set(chain.from_iterable(args.referenceId))
                        if args.referenceId else None)

        return cls(
            args.samfile,
            filterRead=filterRead,
            referenceIds=referenceIds,
            storeQueryIds=storeQueryIds,
            dropUnmapped=args.dropUnmapped,
            dropSecondary=args.dropSecondary,
            dropSupplementary=args.dropSupplementary,
            dropDuplicates=args.dropDuplicates,
            keepQCFailures=args.keepQCFailures,
            minScore=args.minScore,
            maxScore=args.maxScore)
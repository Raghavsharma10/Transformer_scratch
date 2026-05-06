def makeThumbnail(cls, inputFile, person, format, smaller):
        """
        Make a thumbnail of a mugshot image and store it on disk.

        @param inputFile: The image to thumbnail.
        @type inputFile: C{file}

        @param person: The person this mugshot thumbnail is associated with.
        @type person: L{Person}

        @param format: The format of the data in C{inputFile}.
        @type format: C{str} (e.g. I{jpeg})

        @param smaller: Thumbnails are available in two sizes.  if C{smaller}
        is C{True}, then the thumbnail will be in the smaller of the two
        sizes.
        @type smaller: C{bool}

        @return: path to the thumbnail.
        @rtype: L{twisted.python.filepath.FilePath}
        """
        dirsegs = ['mugshots', str(person.storeID)]
        if smaller:
            dirsegs.insert(1, 'smaller')
            size = cls.smallerSize
        else:
            size = cls.size
        atomicOutputFile = person.store.newFile(*dirsegs)
        makeThumbnail(inputFile, atomicOutputFile, size, format)
        atomicOutputFile.close()
        return atomicOutputFile.finalpath
def makeThumbnail(inputFile, outputFile, thumbnailSize, outputFormat='jpeg'):
    """
    Make a thumbnail of the image stored at C{inputPath}, preserving its
    aspect ratio, and write the result to C{outputPath}.

    @param inputFile: The image file (or path to the file) to thumbnail.
    @type inputFile: C{file} or C{str}

    @param outputFile: The file (or path to the file) to write the thumbnail
    to.
    @type outputFile: C{file} or C{str}

    @param thumbnailSize: The maximum length (in pixels) of the longest side of
    the thumbnail image.
    @type thumbnailSize: C{int}

    @param outputFormat: The C{format} argument to pass to L{Image.save}.
    Defaults to I{jpeg}.
    @type format: C{str}
    """
    if Image is None:
        # throw the ImportError here
        import PIL
    image = Image.open(inputFile)
    # Resize needed?
    if thumbnailSize < max(image.size):
        # Convert bilevel and paletted images to grayscale and RGB respectively;
        # otherwise PIL silently switches to Image.NEAREST sampling.
        if image.mode == '1':
            image = image.convert('L')
        elif image.mode == 'P':
            image = image.convert('RGB')
        image.thumbnail((thumbnailSize, thumbnailSize), Image.ANTIALIAS)
    image.save(outputFile, outputFormat)
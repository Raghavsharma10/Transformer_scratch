def pull_arbitrary(collector, image, **kwargs):
    """Pull an arbitrary image"""
    image_index_of = lambda image: urlparse("https://{0}".format(image)).netloc

    if image.startswith("file://"):
        parsed = urlparse(image)
        filename = parsed.netloc + parsed.path
        if not os.path.exists(filename):
            raise HarpoonError("Provided file doesn't exist!", wanted=image)
        with open(filename) as fle:
            image_indexes = [(line.strip(), image_index_of(line.strip())) for line in fle]
    else:
        image_indexes = [(image, image_index_of(image))]

    authentication = collector.configuration.get("authentication", NotSpecified)
    for index, (image, image_index) in enumerate(image_indexes):
        tag = sb.NotSpecified
        if ":" in image:
            image, tag = image.split(":", 1)

        image = {
              "image_name": image
            , "tag": tag
            , "harpoon": collector.configuration["harpoon"]
            , "commands": ["FROM scratch"]
            , "image_index": image_index
            , "assume_role": NotSpecified
            , "authentication": authentication
            }
        meta = Meta(collector.configuration, []).at("images").at("__arbitrary_{0}__".format(index))
        image = HarpoonSpec().image_spec.normalise(meta, image)
        Syncer().pull(image)
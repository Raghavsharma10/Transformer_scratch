def retrieve(collector, image, artifact, **kwargs):
    """Retrieve a file/folder from an image"""
    if artifact in (None, "", NotSpecified):
        raise BadOption("Please specify what to retrieve using the artifact option")

    if collector.configuration["harpoon"].tag is not NotSpecified:
        image.tag = collector.configuration["harpoon"].tag

    # make sure the image is built
    if os.environ.get("NO_BUILD") is None:
        Builder().make_image(image, collector.configuration["images"])

    content = {
          "conf": image
        , "docker_api": collector.configuration["harpoon"].docker_api
        , "images": collector.configuration["images"]
        , "image": image.image_name_with_tag
        , "path": artifact
        }

    # Get us our gold!
    with ContextBuilder().the_context(content) as fle:
        shutil.copyfile(fle.name, os.environ.get("FILENAME", "./retrieved.tar.gz"))
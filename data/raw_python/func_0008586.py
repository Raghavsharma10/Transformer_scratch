def _workaround_no_vector_images(project):
    """Replace vector images with fake ones."""
    RED = (255, 0, 0)
    PLACEHOLDER = kurt.Image.new((32, 32), RED)
    for scriptable in [project.stage] + project.sprites:
        for costume in scriptable.costumes:
            if costume.image.format == "SVG":
                yield "%s - %s" % (scriptable.name, costume.name)
                costume.image = PLACEHOLDER
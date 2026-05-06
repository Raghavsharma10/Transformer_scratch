def compose(target, root=None):
    """Top level function to create a surface.
    
    :param target: the pygame.Surface to blit on. Or a (width, height) tuple
        in which case a new surface will be created

    :type target: -
    """
    if type(root) == Surface:
        raise ValueError("A Surface may not be used as root, please add "
                        +"it as a single child i.e. compose(...)(Surface(...))")
    @_inner_func_anot
    def inner_compose(*children):
        if root:
            root_context = root(*children)
        else:
            assert len(children) == 1
            root_context = children[0]

        if type(target) == pygame.Surface:
            surface = target
            size = target.get_size()
        else:
            size = target
            surface = pygame.Surface(size)

        root_context._draw(surface, pygame.Rect(0, 0, *size))
        return surface
    return inner_compose
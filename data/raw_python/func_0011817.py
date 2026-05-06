def apply_color_map(name: str, mat: np.ndarray = None):
    """returns an RGB matrix scaled by a matplotlib color map"""
    def apply_map(mat):
        return (cm.get_cmap(name)(_normalize(mat))[:, :, :3] * 255).astype(np.uint8)
        
    return apply_map if mat is None else apply_map(mat)
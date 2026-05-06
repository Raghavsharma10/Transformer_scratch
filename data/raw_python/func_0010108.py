def from_lab(l, a, b, alpha=1.0, wref=_DEFAULT_WREF):
    """Create a new instance based on the specifed CIE-LAB values.

    Parameters:
      :l:
        The L component [0...100]
      :a:
        The a component [-1...1]
      :b:
        The a component [-1...1]
      :alpha:
        The color transparency [0...1], default is opaque
      :wref:
        The whitepoint reference, default is 2° D65.

    Returns:
      A grapefruit.Color instance.

    >>> Color.from_lab(66.951823, 0.43084105, 0.73969231)
    Color(1.0, 0.5, -0.0, 1.0)
    >>> Color.from_lab(66.951823, 0.41165967, 0.67282012, wref=WHITE_REFERENCE['std_D50'])
    Color(1.0, 0.5, -0.0, 1.0)
    >>> Color.from_lab(66.951823, 0.43084105, 0.73969231, 0.5)
    Color(1.0, 0.5, -0.0, 0.5)
    >>> Color.from_lab(66.951823, 0.41165967, 0.67282012, 0.5, WHITE_REFERENCE['std_D50'])
    Color(1.0, 0.5, -0.0, 0.5)

    """
    return Color(xyz_to_rgb(*lab_to_xyz(l, a, b, wref)), 'rgb', alpha, wref)
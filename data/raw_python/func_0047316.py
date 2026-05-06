def condense_multidimensional_zeros(css):
    """Replace `:0 0 0 0;`, `:0 0 0;` etc. with `:0;`."""

    css = css.replace(":0 0 0 0;", ":0;")
    css = css.replace(":0 0 0;", ":0;")
    css = css.replace(":0 0;", ":0;")

    # Revert `background-position:0;` to the valid `background-position:0 0;`.
    css = css.replace("background-position:0;", "background-position:0 0;")

    return css
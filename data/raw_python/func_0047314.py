def remove_unnecessary_whitespace(css):
    """Remove unnecessary whitespace characters."""

    def pseudoclasscolon(css):

        """
        Prevents 'p :link' from becoming 'p:link'.

        Translates 'p :link' into 'p ___PSEUDOCLASSCOLON___link'; this is
        translated back again later.
        """

        regex = re.compile(r"(^|\})(([^\{\:])+\:)+([^\{]*\{)")
        match = regex.search(css)
        while match:
            css = ''.join([
                css[:match.start()],
                match.group().replace(":", "___PSEUDOCLASSCOLON___"),
                css[match.end():]])
            match = regex.search(css)
        return css

    css = pseudoclasscolon(css)
    # Remove spaces from before things.
    css = re.sub(r"\s+([!{};:>+\(\)\],])", r"\1", css)

    # If there is a `@charset`, then only allow one, and move to the beginning.
    css = re.sub(r"^(.*)(@charset \"[^\"]*\";)", r"\2\1", css)
    css = re.sub(r"^(\s*@charset [^;]+;\s*)+", r"\1", css)

    # Put the space back in for a few cases, such as `@media screen` and
    # `(-webkit-min-device-pixel-ratio:0)`.
    css = re.sub(r"\band\(", "and (", css)

    # Put the colons back.
    css = css.replace('___PSEUDOCLASSCOLON___', ':')

    # Remove spaces from after things.
    css = re.sub(r"([!{}:;>+\(\[,])\s+", r"\1", css)

    return css
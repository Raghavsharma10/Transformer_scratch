def _chosen_css(self):
        """Read the minified CSS file including STATIC_URL in the references
        to the sprite images."""
        css = render_to_string(self.css_template, {})
        for sprite in self.chosen_sprites:  # rewrite path to sprites in the css
            css = css.replace(sprite, settings.STATIC_URL + "img/" + sprite)
        return css
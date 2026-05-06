def iter_scripts(scratch):
        """A generator for all scripts contained in a scratch file.

        yields stage scripts first, then scripts for each sprite

        """
        for script in scratch.stage.scripts:
            if not isinstance(script, kurt.Comment):
                yield script
        for sprite in scratch.sprites:
            for script in sprite.scripts:
                if not isinstance(script, kurt.Comment):
                    yield script
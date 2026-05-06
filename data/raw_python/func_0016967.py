def update(self, name, color):
        """Update this label.

        :param str name: (required), new name of the label
        :param str color: (required), color code, e.g., 626262, no leading '#'
        :returns: bool
        """
        json = None

        if name and color:
            if color[0] == '#':
                color = color[1:]
            json = self._json(self._patch(self._api, data=dumps({
                'name': name, 'color': color})), 200)

        if json:
            self._update_(json)
            return True

        return False
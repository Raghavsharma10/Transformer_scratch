def set(self, class_name, state, key, value):
        """Set a single style value for a view class and state.

        class_name

            The name of the class to be styled; do not
            include the package name; e.g. 'Button'.

        state

            The name of the state to be stylized. One of the
            following: 'normal', 'focused', 'selected', 'disabled'
            is common.

        key

            The style attribute name; e.g. 'background_color'.

        value

            The value of the style attribute; colors are either
            a 3-tuple for RGB, a 4-tuple for RGBA, or a pair
            thereof for a linear gradient.

        """
        self._styles.setdefault(class_name, {}).setdefault(state, {})
        self._styles[class_name][state][key] = value
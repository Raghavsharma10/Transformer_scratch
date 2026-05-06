def get_dict_for_class(self, class_name, state=None, base_name='View'):
        """The style dict for a given class and state.

        This collects the style attributes from parent classes
        and the class of the given object and gives precedence
        to values thereof to the children.

        The state attribute of the view instance is taken as
        the current state if state is None.

        If the state is not 'normal' then the style definitions
        for the 'normal' state are mixed-in from the given state
        style definitions, giving precedence to the non-'normal'
        style definitions.

        """
        classes = []
        klass = class_name

        while True:
            classes.append(klass)
            if klass.__name__ == base_name:
                break
            klass = klass.__bases__[0]

        if state is None:
            state = 'normal'

        style = {}

        for klass in classes:
            class_name = klass.__name__

            try:
                state_styles = self._styles[class_name][state]
            except KeyError:
                state_styles = {}

            if state != 'normal':
                try:
                    normal_styles = self._styles[class_name]['normal']
                except KeyError:
                    normal_styles = {}

                state_styles = dict(chain(normal_styles.iteritems(),
                                          state_styles.iteritems()))

            style = dict(chain(state_styles.iteritems(),
                               style.iteritems()))

        return style
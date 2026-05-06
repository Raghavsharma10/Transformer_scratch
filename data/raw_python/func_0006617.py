def cree_widgets(self):
        """Create widgets and store them in self.widgets"""
        for t in self.FIELDS:
            if type(t) is str:
                attr, kwargs = t, {}
            else:
                attr, kwargs = t[0], t[1].copy()
            self.champs.append(attr)
            is_editable = kwargs.pop("is_editable", self.is_editable)
            args = [self.acces[attr], is_editable]
            with_base = kwargs.pop("with_base", False)
            if with_base:
                args.append(self.acces.base)

            if 'with_label' in kwargs:
                label = kwargs.pop('with_label')
            else:
                label = ASSOCIATION[attr][0]
            if kwargs:
                w = ASSOCIATION[attr][3](*args, **kwargs)
            else:
                w = ASSOCIATION[attr][3](*args)

            self.widgets[attr] = (w, label)
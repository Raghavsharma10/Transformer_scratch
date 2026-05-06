def _contribute_to_class(self, mcs_args: McsArgs):
        """
        Where the magic happens. Takes one parameter, the :class:`McsArgs` of the
        class-under-construction, and processes the declared ``class Meta`` from
        it (if any). We fill ourself with the declared meta options' name/value pairs,
        give the declared meta options a chance to also contribute to the class-under-
        construction, and finally replace the class-under-construction's ``class Meta``
        with this populated factory instance (aka ``self``).
        """
        self._mcs_args = mcs_args

        Meta = mcs_args.clsdict.pop('Meta', None)  # type: Type[object]
        base_classes_meta = mcs_args.getattr('Meta', None)  # type: MetaOptionsFactory

        mcs_args.clsdict['Meta'] = self  # must come before _fill_from_meta, because
                                         # some meta options may depend upon having
                                         # access to the values of earlier meta options
        self._fill_from_meta(Meta, base_classes_meta, mcs_args)

        for option in self._get_meta_options():
            option_value = getattr(self, option.name, None)
            option.contribute_to_class(mcs_args, option_value)
def _read_var_data(self):
        """ Iterate over the block of this bpch file and return handlers
        in the form of `BPCHDataBundle`s for access to the data contained
        therein.

        """

        var_bundles = OrderedDict()
        var_attrs = OrderedDict()

        n_vars = 0

        while self.fp.tell() < self.fsize:

            var_attr = OrderedDict()

            # read first and second header lines
            line = self.fp.readline('20sffii')
            modelname, res0, res1, halfpolar, center180 = line

            line = self.fp.readline('40si40sdd40s7i')
            category_name, number, unit, tau0, tau1, reserved = line[:6]
            dim0, dim1, dim2, dim3, dim4, dim5, skip = line[6:]
            var_attr['number'] = number

            # Decode byte-strings to utf-8
            category_name = str(category_name, 'utf-8')
            var_attr['category'] = category_name.strip()
            unit = str(unit, 'utf-8')

            # get additional metadata from tracerinfo / diaginfo
            try:
                cat_df = self.diaginfo_df[
                    self.diaginfo_df.name == category_name.strip()
                ]
                # TODO: Safer logic for handling case where more than one
                #       tracer metadata match was made
                # if len(cat_df > 1):
                #     raise ValueError(
                #         "More than one category matching {} found in "
                #         "diaginfo.dat".format(
                #             category_name.strip()
                #         )
                #     )
                # Safe now to select the only row in the DataFrame
                cat = cat_df.T.squeeze()

                tracer_num = int(cat.offset) + int(number)
                diag_df = self.tracerinfo_df[
                    self.tracerinfo_df.tracer == tracer_num
                ]
                # TODO: Safer logic for handling case where more than one
                #       tracer metadata match was made
                # if len(diag_df > 1):
                #     raise ValueError(
                #         "More than one tracer matching {:d} found in "
                #         "tracerinfo.dat".format(tracer_num)
                #     )
                # Safe now to select only row in the DataFrame
                diag = diag_df.T.squeeze()
                diag_attr = diag.to_dict()

                if not unit.strip():  # unit may be empty in bpch
                    unit = diag_attr['unit']  # but not in tracerinfo
                var_attr.update(diag_attr)
            except:
                diag = {'name': '', 'scale': 1}
                var_attr.update(diag)
            var_attr['unit'] = unit

            vname = diag['name']
            fullname = category_name.strip() + "_" + vname

            # parse metadata, get data or set a data proxy
            if dim2 == 1:
                data_shape = (dim0, dim1)         # 2D field
            else:
                data_shape = (dim0, dim1, dim2)
            var_attr['original_shape'] = data_shape

            # Add proxy time dimension to shape
            data_shape = tuple([1, ] + list(data_shape))
            origin = (dim3, dim4, dim5)
            var_attr['origin'] = origin

            timelo, timehi = cf.tau2time(tau0), cf.tau2time(tau1)

            pos = self.fp.tell()
            # Note that we don't pass a dtype, and assume everything is
            # single-fp floats with the correct endian, as hard-coded
            var_bundle = BPCHDataBundle(
                data_shape, self.endian, self.filename, pos, [timelo, timehi],
                metadata=var_attr,
                use_mmap=self.use_mmap, dask_delayed=self.dask_delayed
            )
            self.fp.skipline()

            # Save the data as a "bundle" for concatenating in the final step
            if fullname in var_bundles:
                var_bundles[fullname].append(var_bundle)
            else:
                var_bundles[fullname] = [var_bundle, ]
                var_attrs[fullname] = var_attr
                n_vars += 1

        self.var_data = var_bundles
        self.var_attrs = var_attrs
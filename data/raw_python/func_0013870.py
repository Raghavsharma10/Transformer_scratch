def _load_data(self, date=None, fid=None):
        """
        Load data for an instrument on given date or fid, dependng upon input.

        Parameters
        ------------
        date : (dt.datetime.date object or NoneType)
            file date
        fid : (int or NoneType)
            filename index value

        Returns
        --------
        data : (pds.DataFrame)
            pysat data
        meta : (pysat.Meta)
            pysat meta data
        """

        if fid is not None:
            # get filename based off of index value
            fname = self.files[fid:fid+1]
        elif date is not None:
            fname = self.files[date: date+pds.DateOffset(days=1)]
        else:
            raise ValueError('Must supply either a date or file id number.')
   
        if len(fname) > 0:    
            load_fname = [os.path.join(self.files.data_path, f) for f in fname]
            data, mdata = self._load_rtn(load_fname, tag=self.tag,
                                         sat_id=self.sat_id, **self.kwargs)

            # ensure units and name are named consistently in new Meta
            # object as specified by user upon Instrument instantiation
            mdata.accept_default_labels(self)

        else:
            data = DataFrame(None)
            mdata = _meta.Meta(units_label=self.units_label, name_label=self.name_label,
                        notes_label = self.notes_label, desc_label = self.desc_label,
                        plot_label = self.plot_label, axis_label = self.axis_label,
                        scale_label = self.scale_label, min_label = self.min_label,
                        max_label = self.max_label, fill_label=self.fill_label)

        output_str = '{platform} {name} {tag} {sat_id}'
        output_str = output_str.format(platform=self.platform,
                                       name=self.name, tag=self.tag, 
                                       sat_id=self.sat_id)
        if not data.empty: 
            if not isinstance(data, DataFrame):
                raise TypeError(' '.join(('Data returned by instrument load',
                                'routine must be a pandas.DataFrame')))
            if not isinstance(mdata, _meta.Meta):
                raise TypeError('Metadata returned must be a pysat.Meta object')
            if date is not None:
                output_str = ' '.join(('Returning', output_str, 'data for',
                                       date.strftime('%x')))
            else:
                if len(fname) == 1:
                    # this check was zero
                    output_str = ' '.join(('Returning', output_str, 'data from',
                                           fname[0]))
                else:
                    output_str = ' '.join(('Returning', output_str, 'data from',
                                           fname[0], '::', fname[-1]))
        else:
            # no data signal
            output_str = ' '.join(('No', output_str, 'data for',
                                   date.strftime('%m/%d/%y')))
        # remove extra spaces, if any
        output_str = " ".join(output_str.split())
        print (output_str)                
        return data, mdata
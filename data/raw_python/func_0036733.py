def export(self, **kwargs):
        """
        Generate audio file from composition.

        :param str. filename: Output filename (no extension)
        :param str. filetype: Output file type (only .wav supported for now)
        :param integer samplerate: Sample rate of output audio
        :param integer channels: Channels in output audio, if different than originally specified
        :param bool. separate_tracks: Also generate audio file for each track in composition
        :param int min_length: Minimum length of output array (in frames). Will zero pad extra length.
        :param bool. adjust_dynamics: Automatically adjust dynamics (will document later)

        """
        # get optional args
        filename = kwargs.pop('filename', 'out')
        filetype = kwargs.pop('filetype', 'wav')
        adjust_dynamics = kwargs.pop('adjust_dynamics', False)
        samplerate = kwargs.pop('samplerate', None)
        channels = kwargs.pop('channels', self.channels)
        separate_tracks = kwargs.pop('separate_tracks', False)
        min_length = kwargs.pop('min_length', None)
        
        if samplerate is None:
            samplerate = np.min([track.samplerate for track in self.tracks])

        encoding = 'pcm16'
        to_mp3 = False
        if filetype == 'ogg':
            encoding = 'vorbis'
        elif filetype == 'mp3':
            filetype = 'wav'
            to_mp3 = True

        if separate_tracks:
            # build the separate parts of the composition if desired
            for track in self.tracks:
                out = self.build(track_list=[track],
                                 adjust_dynamics=adjust_dynamics,
                                 min_length=min_length,
                                 channels=channels)
                out_file = Sndfile("%s-%s.%s" %
                                   (filename, track.name, filetype),
                                   'w',
                                   Format(filetype, encoding=encoding),
                                   channels, samplerate)
                out_file.write_frames(out)
                out_file.close()

        # always build the complete composition
        out = self.build(adjust_dynamics=adjust_dynamics,
                         min_length=min_length,
                         channels=channels)

        out_filename = "%s.%s" % (filename, filetype)
        out_file = Sndfile(out_filename, 'w',
                           Format(filetype, encoding=encoding), 
                           channels, samplerate)
        out_file.write_frames(out)
        out_file.close()

        if LIBXMP and filetype == "wav":
            xmp = libxmp.XMPMeta()
            ns = libxmp.consts.XMP_NS_DM
            p = xmp.get_prefix_for_namespace(ns)
            xpath = p + 'Tracks'
            xmp.append_array_item(ns, xpath, None, 
                array_options={"prop_value_is_array": True},
                prop_value_is_struct=True)

            xpath += '[1]/' + p
            xmp.set_property(ns, xpath + "trackName", "CuePoint Markers")
            xmp.set_property(ns, xpath + "trackType", "Cue")
            xmp.set_property(ns, xpath + "frameRate", "f%d" % samplerate)

            for i, lab in enumerate(self.labels):
                xmp.append_array_item(ns, xpath + "markers", None,
                    array_options={"prop_value_is_array": True},
                    prop_value_is_struct=True)
                xmp.set_property(ns,
                    xpath + "markers[%d]/%sname" % (i + 1, p), lab.name)
                xmp.set_property(ns,
                    xpath + "markers[%d]/%sstartTime" % (i + 1, p),
                    str(lab.sample(samplerate))) 

            xmpfile = libxmp.XMPFiles(file_path=out_filename, open_forupdate=True)
            if xmpfile.can_put_xmp(xmp):
                xmpfile.put_xmp(xmp)
            xmpfile.close_file()

        if to_mp3:
            wav_to_mp3(out_filename, delete_wav=True)

        return out
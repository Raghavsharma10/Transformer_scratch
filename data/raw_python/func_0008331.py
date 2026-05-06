def update(self):
        """
        Updates run & tel status window. Runs
        once every 2 seconds.
        """
        g = get_root(self).globals

        if g.astro is None or g.fpslide is None:
            self.after(100, self.update)
            return

        try:
            if g.cpars['tcs_on']:

                try:
                    # Poll TCS for ra,dec etc.
                    ra, dec, pa, focus = self.tcs_data_queue.get(block=False)

                    # format ra, dec as HMS
                    coo = coord.SkyCoord(ra, dec, unit=(u.deg, u.deg))
                    ratxt = coo.ra.to_string(sep=':', unit=u.hour, precision=0)
                    dectxt = coo.dec.to_string(sep=':', unit=u.deg,
                                               alwayssign=True,
                                               precision=0)
                    self.ra.configure(text=ratxt)
                    self.dec.configure(text=dectxt)

                    # wrap pa from 0 to 360
                    pa = coord.Longitude(pa*u.deg)
                    self.pa.configure(text='{0:6.2f}'.format(pa.value))

                    # set focus
                    self.focus.configure(text='{0:+5.2f}'.format(focus))

                    # Calculate most of the
                    # stuff that we don't get from the telescope
                    now = Time.now()
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        # ignore astropy deprecation warnings
                        lon = g.astro.obs.longitude
                    lst = now.sidereal_time(kind='mean',
                                            longitude=lon)
                    ha = coo.ra.hourangle*u.hourangle - lst
                    hatxt = ha.wrap_at(12*u.hourangle).to_string(sep=':', precision=0)
                    self.ha.configure(text=hatxt)

                    altaz_frame = coord.AltAz(obstime=now, location=g.astro.obs)
                    altaz = coo.transform_to(altaz_frame)
                    self.alt.configure(text='{0:<4.1f}'.format(altaz.alt.value))
                    self.az.configure(text='{0:<5.1f}'.format(altaz.az.value))
                    # set airmass
                    self.airmass.configure(text='{0:<4.2f}'.format(altaz.secz))

                    # distance to the moon. Warn if too close
                    # (configurable) to it.
                    md = coord.get_moon(now, g.astro.obs).separation(coo)
                    self.mdist.configure(text='{0:<7.2f}'.format(md.value))
                    if md < g.cpars['mdist_warn']*u.deg:
                        self.mdist.configure(bg=g.COL['warn'])
                    else:
                        self.mdist.configure(bg=g.COL['main'])
                except Empty:
                    # silently do nothing if queue is empty
                    pass
                except Exception as err:
                    self.ra.configure(text='UNDEF')
                    self.dec.configure(text='UNDEF')
                    self.pa.configure(text='UNDEF')
                    self.ha.configure(text='UNDEF')
                    self.alt.configure(text='UNDEF')
                    self.az.configure(text='UNDEF')
                    self.airmass.configure(text='UNDEF')
                    self.mdist.configure(text='UNDEF')
                    g.clog.warn('TCS error: ' + str(err))

            if g.cpars['hcam_server_on'] and \
               g.cpars['eso_server_online']:

                # get run number (set by the 'Start' button')
                try:
                    # get run number from hipercam server
                    run = getRunNumber(g)
                    self.run.configure(text='{0:03d}'.format(run))

                    # Find the number of frames in this run
                    try:
                        frame_no = getFrameNumber(g)
                        self.frame.configure(text='{0:04d}'.format(frame_no))
                    except Exception as err:
                        if err.code == 404:
                            self.frame.configure(text='0')
                        else:
                            g.clog.debug('Error occurred trying to set frame')
                            self.frame.configure(text='UNDEF')

                except Exception as err:
                    g.clog.debug('Error trying to set run: ' + str(err))

            # get the slide position
            # poll at 5x slower rate than the frame
            if self.count % 5 == 0 and g.cpars['focal_plane_slide_on']:
                try:
                    pos_ms, pos_mm, pos_px = self.slide_pos_queue.get(block=False)
                    self.fpslide.configure(text='{0:d}'.format(
                        int(round(pos_px))))
                    if pos_px < 1050.:
                        self.fpslide.configure(bg=g.COL['warn'])
                    else:
                        self.fpslide.configure(bg=g.COL['main'])
                except Exception as err:
                    pass

            # get the CCD temperature poll at 5x slower rate than the frame
            if self.count % 5 == 0:
                try:
                    if g.ccd_hw is not None and g.ccd_hw.ok:
                        self.ccd_temps.configure(text='OK')
                        self.ccd_temps.configure(bg=g.COL['main'])
                    else:
                        self.ccd_temps.configure(text='ERR')
                        self.ccd_temps.configure(bg=g.COL['warn'])
                except Exception as err:
                    g.clog.warn(str(err))
                    self.ccd_temps.configure(text='UNDEF')
                    self.ccd_temps.configure(bg=g.COL['warn'])

        except Exception as err:
            # this is a safety catchall trap as it is important
            # that this routine keeps going
            g.clog.warn('Unexpected error: ' + str(err))

        # run every 2 seconds
        self.count += 1
        self.after(2000, self.update)
def check(self, *args):
        """
        Callback to check validity of instrument parameters.

        Performs the following tasks:
            - spots and flags overlapping windows or null window parameters
            - flags windows with invalid dimensions given the binning parameter
            - sets the correct number of enabled windows
            - disables or enables clear and nod buttons depending on drift mode or not
            - checks for window synchronisation, enabling sync button if required
            - enables or disables start button if settings are OK

        Returns
        -------
        status : bool
            True or False according to whether the settings are OK.
        """
        status = True
        g = get_root(self).globals

        # clear errors on binning (may be set later if FF)
        xbinw, ybinw = self.wframe.xbin, self.wframe.ybin
        xbinw.config(bg=g.COL['main'])
        ybinw.config(bg=g.COL['main'])

        # keep binning factors of drift mode and windowed mode up to date
        oframe, aframe = ((self.quad_frame, self.drift_frame) if self.drift_frame.winfo_ismapped()
                          else (self.drift_frame, self.quad_frame))
        xbin, ybin = aframe.xbin.value(), aframe.ybin.value()
        oframe.xbin.set(xbin)
        oframe.ybin.set(ybin)

        if not self.frozen:
            if self.clear() or self.isDrift():
                # disable nmult in clear or drift mode
                self.nmult.disable()
            else:
                self.nmult.enable()

        if self.isDrift():
            self.clearLab.config(state='disable')
            self.nodLab.config(state='disable')
            if not self.drift_frame.winfo_ismapped():
                self.quad_frame.grid_forget()
                self.drift_frame.grid(row=10, column=0, columnspan=3,
                                      sticky=tk.W+tk.N)

            if not self.frozen:
                self.oscany.config(state='disable')
                self.oscan.config(state='disable')
                self.clear.config(state='disable')
                self.nod.config(state='disable')
                self.wframe.enable()
                status = self.wframe.check()

        elif self.isFF():
            # special case check of binning from window frame
            if 1024 % xbin != 0:
                status = False
                xbinw.config(bg=g.COL['error'])
            elif (1024 // xbin) % 4 != 0:
                status = False
                xbinw.config(bg=g.COL['error'])
            if 512 % ybin != 0:
                status = False
                ybinw.config(bg=g.COL['error'])

            if not self.quad_frame.winfo_ismapped():
                self.drift_frame.grid_forget()
                self.quad_frame.grid(row=10, column=0, columnspan=3,
                                     sticky=tk.W+tk.N)

            self.clearLab.config(state='normal')
            if g.cpars['telins_name'] == 'GTC':
                self.nodLab.config(state='normal')
            else:
                self.nodLab.config(state='disable')
            if not self.frozen:
                self.oscany.config(state='normal')
                self.oscan.config(state='normal')
                self.clear.config(state='normal')
                if g.cpars['telins_name'] == 'GTC':
                    self.nod.config(state='normal')
                else:
                    self.nod.config(state='disable')
                self.wframe.disable()

        else:
            self.clearLab.config(state='normal')
            if g.cpars['telins_name'] == 'GTC':
                self.nodLab.config(state='normal')
            else:
                self.nodLab.config(state='disable')
            if not self.quad_frame.winfo_ismapped():
                self.drift_frame.grid_forget()
                self.quad_frame.grid(row=10, column=0, columnspan=3,
                                     sticky=tk.W+tk.N)

            if not self.frozen:
                self.oscany.config(state='disable')
                self.oscan.config(state='normal')
                self.clear.config(state='normal')
                if g.cpars['telins_name'] == 'GTC':
                    self.nod.config(state='normal')
                else:
                    self.nod.config(state='disable')
                self.wframe.enable()
                status = self.wframe.check()

        # exposure delay
        if self.expose.ok():
            self.expose.config(bg=g.COL['main'])
        else:
            self.expose.config(bg=g.COL['warn'])
            status = False

        # don't allow binning other than 1, 2 in overscan or prescan mode
        if self.oscan() or self.oscany():
            if xbin not in (1, 2):
                status = False
                xbinw.config(bg=g.COL['error'])
            if ybin not in (1, 2):
                status = False
                ybinw.config(bg=g.COL['error'])

        # disable clear if nodding enabled. re-enable if not drift
        if not self.frozen:
            if self.nod() or self.nodPattern:
                self.clear.config(state='disabled')
                self.clearLab.config(state='disabled')
            elif not self.isDrift():
                self.clear.config(state='normal')
                self.clearLab.config(state='normal')

        # allow posting if parameters are OK. update count and SN estimates too
        if status:
            if (g.cpars['hcam_server_on'] and g.cpars['eso_server_online'] and
                    g.observe.start['state'] == 'disabled' and
                    not isRunActive(g)):
                g.observe.start.enable()
            g.count.update()
        else:
            g.observe.start.disable()

        return status
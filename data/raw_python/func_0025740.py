def syncParamList(self, firstTime, preserve_order=True):
        """ Set or reset the internal param list from the dict's contents. """
        # See the note in setParam about this design.

        # Get latest par values from dict.  Make sure we do not
        # change the id of the __paramList pointer here.
        new_list = self._getParamsFromConfigDict(self, initialPass=firstTime)
                                               # dumpCfgspcTo=sys.stdout)
        # Have to add this odd last one for the sake of the GUI (still?)
        if self._forUseWithEpar:
            new_list.append(basicpar.IrafParS(['$nargs','s','h','N']))

        if len(self.__paramList) > 0 and preserve_order:
            # Here we have the most up-to-date data from the actual data
            # model, the ConfigObj dict, and we need to use it to fill in
            # our param list.  BUT, we need to preserve the order our list
            # has had up until now (by unique parameter name).
            namesInOrder = [p.fullName() for p in self.__paramList]
            assert len(namesInOrder) == len(new_list), \
                   'Mismatch in num pars, had: '+str(len(namesInOrder))+ \
                   ', now we have: '+str(len(new_list))+', '+ \
                   str([p.fullName() for p in new_list])
            self.__paramList[:] = [] # clear list, keep same pointer
            # create a flat dict view of new_list, for ease of use in next step
            new_list_dict = {} # can do in one step in v2.7
            for par in new_list: new_list_dict[par.fullName()] = par
            # populate
            for fn in namesInOrder:
                self.__paramList.append(new_list_dict[fn])
        else:
            # Here we just take the data in whatever order it came.
            self.__paramList[:] = new_list
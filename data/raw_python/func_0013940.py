def _apply_all(self, sat):
        """
        Apply all of the custom functions to the satellite data object.
        """
        if len(self._functions) > 0:
            for func, arg, kwarg, kind in zip(self._functions, self._args, 
                                                self._kwargs, self._kind):
                if len(sat.data) > 0:     
                    if kind == 'add':
                        # apply custom functions that add data to the
                        # instrument object
                        tempd = sat.copy()
                        newData = func(tempd, *arg, **kwarg)
                        del tempd

                        # process different types of data returned by the
                        # function if a dict is returned, data in 'data'
                        if isinstance(newData,dict):
                            # if DataFrame returned, add Frame to existing frame
                            if isinstance(newData['data'], pds.DataFrame):
                                sat[newData['data'].columns] = newData
                            # if a series is returned, add it as a column
                            elif isinstance(newData['data'], pds.Series):
                                # look for name attached to series first
                                if newData['data'].name is not None:
                                    sat[newData['data'].name] = newData
                                # look if name is provided as part of dict
                                # returned from function
                                elif 'name' in newData.keys():
                                    name = newData.pop('name')
                                    sat[name] = newData
                                # couldn't find name information
                                else:
                                    raise ValueError('Must assign a name to ' +
                                                     'Series or return a ' +
                                                     '"name" in dictionary.')

                            # some kind of iterable was returned
                            elif hasattr(newData['data'], '__iter__'):
                                # look for name in returned dict
                                if 'name' in newData.keys():
                                    name = newData.pop('name')
                                    sat[name] = newData
                                else:
                                    raise ValueError('Must include "name" in ' +
                                                     'returned dictionary.')

                        # bare DataFrame is returned
                        elif isinstance(newData, pds.DataFrame):
                            sat[newData.columns] = newData
                        # bare Series is returned, name must be attached to
                        # Series
                        elif isinstance(newData, pds.Series):
                            sat[newData.name] = newData      

                        # some kind of iterable returned,
                        # presuming (name, data)
                        # or ([name1,...], [data1,...])                      
                        elif hasattr(newData, '__iter__'):
                            # falling back to older behavior
                            # unpack tuple/list that was returned
                            newName = newData[0]
                            newData = newData[1]
                            if len(newData)>0:
                                # doesn't really check ensure data, there could
                                # be multiple empty arrays returned, [[],[]]
                                if isinstance(newName, str):
                                    # one item to add
                                    sat[newName] = newData
                                else:    		
                                    # multiple items
                                    for name, data in zip(newName, newData):
                                        if len(data)>0:        
                                            # fixes up the incomplete check
                                            # from before
                                            sat[name] = data
                        else:
                            raise ValueError("kernel doesn't know what to do " +
                                             "with returned data.")

                    # modifying loaded data
                    if kind == 'modify':
                        t = func(sat,*arg,**kwarg)
                        if t is not None:
                            raise ValueError('Modify functions should not ' +
                                             'return any information via ' +
                                             'return. Information may only be' +
                                             ' propagated back by modifying ' +
                                             'supplied pysat object.')

                    # pass function (function runs, no data allowed back)
                    if kind == 'pass':
                        tempd = sat.copy()
                        t = func(tempd,*arg,**kwarg)
                        del tempd
                        if t is not None:
                            raise ValueError('Pass functions should not ' +
                                             'return any information via ' +
                                             'return.')
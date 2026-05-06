def predict(self, query_data, verbose=False, distribution=False, cleanup=True):
        """
        Iterates over the predicted values and probability (if supported).
        Each iteration yields a tuple of the form (prediction, probability).
        
        If the file is a test file (i.e. contains no query variables),
        then the tuple will be of the form (prediction, actual).
        
        See http://weka.wikispaces.com/Making+predictions
        for further explanation on interpreting Weka prediction output.
        """
        model_fn = None
        query_fn = None
        clean_query = False
        stdout = None
        try:
            
            # Validate query data.
            if isinstance(query_data, basestring):
                assert os.path.isfile(query_data)
                query_fn = query_data
            else:
                #assert isinstance(query_data, arff.ArffFile) #TODO: doesn't work in Python 3.*?
                assert type(query_data).__name__ == 'ArffFile', 'Must be of type ArffFile, not "%s"' % type(query_data).__name__
                fd, query_fn = tempfile.mkstemp(suffix='.arff')
                if verbose:
                    print('writing', query_fn)
                os.close(fd)
                open(query_fn, 'w').write(query_data.write())
                clean_query = True
            assert query_fn
                
            # Validate model file.
            fd, model_fn = tempfile.mkstemp()
            os.close(fd)
            assert self._model_data, "You must train this classifier before predicting."
            fout = open(model_fn, 'wb')
            fout.write(self._model_data)
            fout.close()
            
#            print(open(model_fn).read()
#            print(open(query_fn).read()
            # Call Weka Jar.
            args = dict(
                CP=CP,
                classifier_name=self.name,
                model_fn=model_fn,
                query_fn=query_fn,
                #ckargs = self._get_ckargs_str(),
                distribution=('-distribution' if distribution else ''),
            )
            cmd = ("java -cp %(CP)s %(classifier_name)s -p 0 %(distribution)s -l \"%(model_fn)s\" -T \"%(query_fn)s\"") % args
            if verbose:
                print(cmd)
            p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True)
            stdin, stdout, stderr = (p.stdin, p.stdout, p.stderr)
            stdout_str = stdout.read()
            stderr_str = stderr.read()
            if verbose:
                print('stdout:')
                print(stdout_str)
                print('stderr:')
                print(stderr_str)
            if stderr_str:
                raise PredictionError(stderr_str)
            
            if stdout_str:
                # inst#     actual  predicted error prediction
                #header = 'inst,actual,predicted,error'.split(',')
                query = arff.ArffFile.load(query_fn)
                query_variables = [
                    query.attributes[i]
                    for i, v in enumerate(query.data[0])
                    if v == arff.MISSING]
                if not query_variables:
                    query_variables = [query.attributes[-1]]
#                assert query_variables, \
#                    "There must be at least one query variable in the query."
                if verbose:
                    print('query_variables:', query_variables)
                header = 'predicted'.split(',')
                # sample line:     1        1:?       4:36   +   1
                
                # Expected output without distribution:
                #=== Predictions on test data ===
                #
                # inst#     actual  predicted error prediction
                #     1        1:? 11:Acer_tr   +   1

                #=== Predictions on test data ===
                #
                # inst#     actual  predicted      error
                #     1          ?      7              ? 

                #=== Predictions on test data ===
                #
                # inst#     actual  predicted error prediction
                #     1        1:?        1:0       0.99 
                #     2        1:?        1:0       0.99 
                #     3        1:?        1:0       0.99 
                #     4        1:?        1:0       0.99 
                #     5        1:?        1:0       0.99 

                # Expected output with distribution:
                #=== Predictions on test data ===
                #
                # inst#     actual  predicted error distribution
                #     1        1:? 11:Acer_tr   +   0,0,0,0,0,0,0,0,0,0,*1,0,0,0,0,0...

                # Expected output with simple format:
                # inst#     actual  predicted      error
                #     1          ?     -3.417          ? 


                q = re.findall(
                    r'J48 pruned tree\s+\-+:\s+([0-9]+)\s+',
                    stdout_str.decode('utf-8'), re.MULTILINE|re.DOTALL)
                if q:
                    class_label = q[0]
                    prob = 1.0
                    yield PredictionResult(
                        actual=None,
                        predicted=class_label,
                        probability=prob,)
                elif re.findall(r'error\s+(?:distribution|prediction)', stdout_str.decode('utf-8')):
                    # Check for distribution output.
                    matches = re.findall(
                        r"^\s*[0-9\.]+\s+[a-zA-Z0-9\.\?\:]+\s+(?P<cls_value>[a-zA-Z0-9_\.\?\:]+)\s+\+?\s+(?P<prob>[a-zA-Z0-9\.\?\,\*]+)",
                        stdout_str.decode('utf-8'),
                        re.MULTILINE)
                    assert matches, ("No results found matching distribution pattern in stdout: %s") % stdout_str
                    for match in matches:
                        prediction, prob = match
                        class_index, class_label = prediction.split(':')
                        class_index = int(class_index)
                        if distribution:
                            # Convert list of probabilities into a hash linking the prob
                            # to the associated class value.
                            prob = dict(zip(
                                query.attribute_data[query.attributes[-1]],
                                map(float, prob.replace('*', '').split(','))))
                        else:
                            prob = float(prob)
                        class_label = query.attribute_data[query.attributes[-1]][class_index-1]
                        yield PredictionResult(
                            actual=None,
                            predicted=class_label,
                            probability=prob,)
                else:
                    # Otherwise, assume a simple output.
                    matches = re.findall(
                        # inst#     actual  predicted 
                        r"^\s*([0-9\.]+)\s+([a-zA-Z0-9\-\.\?\:]+)\s+([a-zA-Z0-9\-_\.\?\:]+)\s+",
                        stdout_str.decode('utf-8'),
                        re.MULTILINE)
                    assert matches, "No results found matching simple pattern in stdout: %s" % stdout_str
                    #print('matches:',len(matches)
                    for match in matches:
                        inst, actual, predicted = match
                        class_name = query.attributes[-1]
                        actual_value = query.get_attribute_value(class_name, actual)
                        predicted_value = query.get_attribute_value(class_name, predicted)
                        yield PredictionResult(
                            actual=actual_value,
                            predicted=predicted_value,
                            probability=None,)
        finally:
            # Cleanup files.
            if cleanup:
                if model_fn:
                    self._model_data = open(model_fn, 'rb').read()
                    os.remove(model_fn)
                if query_fn and clean_query:
                    os.remove(query_fn)
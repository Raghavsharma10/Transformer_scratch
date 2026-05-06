def train(self, training_data, testing_data=None, verbose=False):
        """
        Updates the classifier with new data.
        """
        model_fn = None
        training_fn = None
        clean_training = False
        testing_fn = None
        clean_testing = False
        try:
            
            # Validate training data.
            if isinstance(training_data, basestring):
                assert os.path.isfile(training_data)
                training_fn = training_data
            else:
                assert isinstance(training_data, arff.ArffFile)
                fd, training_fn = tempfile.mkstemp(suffix='.arff')
                os.close(fd)
                with open(training_fn, 'w') as fout:
                    fout.write(training_data.write())
                clean_training = True
            assert training_fn
                
            # Validate testing data.
            if testing_data:
                if isinstance(testing_data, basestring):
                    assert os.path.isfile(testing_data)
                    testing_fn = testing_data
                else:
                    assert isinstance(testing_data, arff.ArffFile)
                    fd, testing_fn = tempfile.mkstemp(suffix='.arff')
                    os.close(fd)
                    with open(testing_fn, 'w') as fout:
                        fout.write(testing_data.write())
                    clean_testing = True
            else:
                testing_fn = training_fn
            assert testing_fn
                
            # Validate model file.
            fd, model_fn = tempfile.mkstemp()
            os.close(fd)
            if self._model_data:
                fout = open(model_fn, 'wb')
                fout.write(self._model_data)
                fout.close()
            
            # Call Weka Jar.
            args = dict(
                CP=CP,
                classifier_name=self.name,
                model_fn=model_fn,
                training_fn=training_fn,
                testing_fn=testing_fn,
                ckargs=self._get_ckargs_str(),
            )
            if self._model_data:
                # Load existing model.
                cmd = (
                    "java -cp %(CP)s %(classifier_name)s -l \"%(model_fn)s\" "
                    "-t \"%(training_fn)s\" -T \"%(testing_fn)s\" -d \"%(model_fn)s\"") % args
            else:
                # Create new model file.
                cmd = (
                    "java -cp %(CP)s %(classifier_name)s -t \"%(training_fn)s\" "
                    "-T \"%(testing_fn)s\" -d \"%(model_fn)s\" %(ckargs)s") % args
            if verbose:
                print(cmd)
            p = Popen(
                cmd,
                shell=True,
                stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=sys.platform != "win32")
            stdin, stdout, stderr = (p.stdin, p.stdout, p.stderr)
            stdout_str = stdout.read()
            stderr_str = stderr.read()
            
            self.last_training_stdout = stdout_str
            self.last_training_stderr = stderr_str
            
            if verbose:
                print('stdout:')
                print(stdout_str)
                print('stderr:')
                print(stderr_str)
            # exclude "Warning" lines not to raise an error for a simple warning
            stderr_str = '\n'.join(l for l in stderr_str.decode('utf8').split('\n') if not "Warning" in l)
            if stderr_str:
                raise TrainingError(stderr_str)
            
            # Save schema.
            if not self.schema:
                self.schema = arff.ArffFile.load(training_fn, schema_only=True).copy(schema_only=True)
            
            # Save model.
            with open(model_fn, 'rb') as fin:
                self._model_data = fin.read()
            assert self._model_data
        finally:
            # Cleanup files.
            if model_fn:
                os.remove(model_fn)
            if training_fn and clean_training:
                os.remove(training_fn)
            if testing_fn and clean_testing:
                os.remove(testing_fn)
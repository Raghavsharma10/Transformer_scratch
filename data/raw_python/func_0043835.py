def _write_submit_script(self, script_string, script_filename):
        '''
        Load the template string with config values and write the generated submit script to
        a submit script file.

        Args:
              - template_string (string) : The template string to be used for the writing submit script
              - script_filename (string) : Name of the submit script

        Returns:
              - True: on success

        Raises:
              SchedulerMissingArgs : If template is missing args
              ScriptPathError : Unable to write submit script out
        '''

        try:
            with open(script_filename, 'w') as f:
                f.write(script_string)

        except KeyError as e:
            logger.error("Missing keys for submit script : %s", e)
            raise (ep_error.SchedulerMissingArgs(e.args, self.label))

        except IOError as e:
            logger.error("Failed writing to submit script: %s", script_filename)
            raise (ep_error.ScriptPathError(script_filename, e))

        return True
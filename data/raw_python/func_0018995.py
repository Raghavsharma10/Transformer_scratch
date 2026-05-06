def init_model(self, clear_registry: bool = True) -> None:
        """Load the control file of the actual |Element| object, initialise
        its |Model| object, build the required connections via (an eventually
        overridden version of) method |Model.connect| of class |Model|, and
        update its  derived parameter values via calling (an eventually
        overridden version) of method |Parameters.update| of class |Parameters|.


        See method |HydPy.init_models| of class |HydPy| and property
        |model| of class |Element| fur further information.
        """
        try:
            with hydpy.pub.options.warnsimulationstep(False):
                info = hydpy.pub.controlmanager.load_file(
                    element=self, clear_registry=clear_registry)
                self.model = info['model']
                self.model.parameters.update()
        except OSError:
            if hydpy.pub.options.warnmissingcontrolfile:
                warnings.warn(
                    f'Due to a missing or no accessible control file, no '
                    f'model could be initialised for element `{self.name}`')
            else:
                objecttools.augment_excmessage(
                    f'While trying to initialise the model '
                    f'object of element `{self.name}`')
        except BaseException:
            objecttools.augment_excmessage(
                f'While trying to initialise the model '
                f'object of element `{self.name}`')
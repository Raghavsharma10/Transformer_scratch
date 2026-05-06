def create_spectrum(spectrum_name):
        """Initialize spectrum and peak descriptions.

        :param str spectrum_name: Name of the spectrum from which peak list will be simulated.
        :return: Spectrum object.
        :rtype: :class:`~nmrstarlib.plsimulator.Spectrum`
        """
        try:
            spectrum_description = nmrstarlib.SPECTRUM_DESCRIPTIONS[spectrum_name]
        except KeyError:
            raise NotImplementedError("Experiment type is not defined.")

        spectrum = plsimulator.Spectrum(spectrum_name, spectrum_description["Labels"],
                                        spectrum_description["MinNumberPeaksPerSpinSystem"],
                                        spectrum_description.get("ResonanceLimit", None))

        for peak_descr in spectrum_description["PeakDescriptions"]:
            spectrum.append(plsimulator.PeakDescription(peak_descr["fraction"], peak_descr["dimensions"]))

        return spectrum
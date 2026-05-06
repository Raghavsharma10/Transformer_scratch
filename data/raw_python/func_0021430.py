def updatePlotWidget(self):
        """Updating the plotting widget should not require any information
        about the current state of the widget."""
        pw = self.getPlotWidget()
        pw.reset()

        results = self.resultsModel.getCheckedItems()

        for result in results:
            if isinstance(result, ExperimentalData):
                spectrum = result.spectra['Expt']
                spectrum.legend = '{}-{}'.format(result.index, 'Expt')
                spectrum.xLabel = 'X'
                spectrum.yLabel = 'Y'
                spectrum.plot(plotWidget=pw)
            else:
                if len(results) > 1 and result.experiment in ['RIXS', ]:
                    continue
                for spectrum in result.spectra.processed:
                    spectrum.legend = '{}-{}'.format(
                        result.index, spectrum.shortName)
                    if spectrum.name in result.spectra.toPlotChecked:
                        spectrum.plot(plotWidget=pw)
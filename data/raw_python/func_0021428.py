def populateWidget(self):
        """
        Populate the widget using data stored in the state
        object. The order in which the individual widgets are populated
        follows their arrangment.

        The models are recreated every time the function is called.
        This might seem to be an overkill, but in practice it is very fast.
        Don't try to move the model creation outside this function; is not
        worth the effort, and there is nothing to gain from it.
        """
        self.elementComboBox.setItems(self.state._elements, self.state.element)
        self.chargeComboBox.setItems(self.state._charges, self.state.charge)
        self.symmetryComboBox.setItems(
            self.state._symmetries, self.state.symmetry)
        self.experimentComboBox.setItems(
            self.state._experiments, self.state.experiment)
        self.edgeComboBox.setItems(self.state._edges, self.state.edge)

        self.temperatureLineEdit.setValue(self.state.temperature)
        self.magneticFieldLineEdit.setValue(self.state.magneticField)

        self.axesTabWidget.setTabText(0, str(self.state.xLabel))
        self.xMinLineEdit.setValue(self.state.xMin)
        self.xMaxLineEdit.setValue(self.state.xMax)
        self.xNPointsLineEdit.setValue(self.state.xNPoints)
        self.xLorentzianLineEdit.setList(self.state.xLorentzian)
        self.xGaussianLineEdit.setValue(self.state.xGaussian)

        self.k1LineEdit.setVector(self.state.k1)
        self.eps11LineEdit.setVector(self.state.eps11)
        self.eps12LineEdit.setVector(self.state.eps12)

        if self.state.experiment in ['RIXS', ]:
            if self.axesTabWidget.count() == 1:
                tab = self.axesTabWidget.findChild(QWidget, 'yTab')
                self.axesTabWidget.addTab(tab, tab.objectName())
                self.axesTabWidget.setTabText(1, self.state.yLabel)
            self.yMinLineEdit.setValue(self.state.yMin)
            self.yMaxLineEdit.setValue(self.state.yMax)
            self.yNPointsLineEdit.setValue(self.state.yNPoints)
            self.yLorentzianLineEdit.setList(self.state.yLorentzian)
            self.yGaussianLineEdit.setValue(self.state.yGaussian)
            self.k2LineEdit.setVector(self.state.k2)
            self.eps21LineEdit.setVector(self.state.eps21)
            self.eps22LineEdit.setVector(self.state.eps22)
            text = self.eps11Label.text()
            text = re.sub('>[vσ]', '>σ', text)
            self.eps11Label.setText(text)
            text = self.eps12Label.text()
            text = re.sub('>[hπ]', '>π', text)
            self.eps12Label.setText(text)
        else:
            self.axesTabWidget.removeTab(1)
            text = self.eps11Label.text()
            text = re.sub('>[vσ]', '>v', text)
            self.eps11Label.setText(text)
            text = self.eps12Label.text()
            text = re.sub('>[hπ]', '>h', text)
            self.eps12Label.setText(text)

        # Create the spectra selection model.
        self.spectraModel = SpectraModel(parent=self)
        self.spectraModel.setModelData(
            self.state.spectra.toCalculate,
            self.state.spectra.toCalculateChecked)
        self.spectraModel.checkStateChanged.connect(
            self.updateSpectraCheckState)
        self.spectraListView.setModel(self.spectraModel)
        self.spectraListView.selectionModel().setCurrentIndex(
            self.spectraModel.index(0, 0), QItemSelectionModel.Select)

        self.fkLineEdit.setValue(self.state.fk)
        self.gkLineEdit.setValue(self.state.gk)
        self.zetaLineEdit.setValue(self.state.zeta)

        # Create the Hamiltonian model.
        self.hamiltonianModel = HamiltonianModel(parent=self)
        self.hamiltonianModel.setModelData(self.state.hamiltonianData)
        self.hamiltonianModel.setNodesCheckState(self.state.hamiltonianState)
        if self.syncParametersCheckBox.isChecked():
            self.hamiltonianModel.setSyncState(True)
        else:
            self.hamiltonianModel.setSyncState(False)
        self.hamiltonianModel.dataChanged.connect(self.updateHamiltonianData)
        self.hamiltonianModel.itemCheckStateChanged.connect(
            self.updateHamiltonianNodeCheckState)

        # Assign the Hamiltonian model to the Hamiltonian terms view.
        self.hamiltonianTermsView.setModel(self.hamiltonianModel)
        self.hamiltonianTermsView.selectionModel().setCurrentIndex(
            self.hamiltonianModel.index(0, 0), QItemSelectionModel.Select)
        self.hamiltonianTermsView.selectionModel().selectionChanged.connect(
            self.selectedHamiltonianTermChanged)

        # Assign the Hamiltonian model to the Hamiltonian parameters view.
        self.hamiltonianParametersView.setModel(self.hamiltonianModel)
        self.hamiltonianParametersView.expandAll()
        self.hamiltonianParametersView.resizeAllColumnsToContents()
        self.hamiltonianParametersView.setColumnWidth(0, 130)
        self.hamiltonianParametersView.setRootIndex(
            self.hamiltonianTermsView.currentIndex())

        self.nPsisLineEdit.setValue(self.state.nPsis)
        self.nPsisAutoCheckBox.setChecked(self.state.nPsisAuto)
        self.nConfigurationsLineEdit.setValue(self.state.nConfigurations)

        self.nConfigurationsLineEdit.setEnabled(False)
        name = '{}-Ligands Hybridization'.format(self.state.block)
        for termName in self.state.hamiltonianData:
            if name in termName:
                termState = self.state.hamiltonianState[termName]
                if termState == 0:
                    continue
                else:
                    self.nConfigurationsLineEdit.setEnabled(True)

        if not hasattr(self, 'resultsModel'):
            # Create the results model.
            self.resultsModel = ResultsModel(parent=self)
            self.resultsModel.itemNameChanged.connect(
                self.updateCalculationName)
            self.resultsModel.itemCheckStateChanged.connect(
                self.updatePlotWidget)
            self.resultsModel.dataChanged.connect(self.updatePlotWidget)
            self.resultsModel.dataChanged.connect(self.updateResultsView)

            # Assign the results model to the results view.
            self.resultsView.setModel(self.resultsModel)
            self.resultsView.selectionModel().selectionChanged.connect(
                self.selectedResultsChanged)
            self.resultsView.resizeColumnsToContents()
            self.resultsView.horizontalHeader().setSectionsMovable(False)
            self.resultsView.horizontalHeader().setSectionsClickable(False)
            if sys.platform == 'darwin':
                self.resultsView.horizontalHeader().setMaximumHeight(17)

            # Add a context menu to the view.
            self.resultsView.setContextMenuPolicy(Qt.CustomContextMenu)
            self.resultsView.customContextMenuRequested[QPoint].connect(
                self.showResultsContextMenu)

        if not hasattr(self, 'resultDetailsDialog'):
            self.resultDetailsDialog = QuantyResultDetailsDialog(parent=self)

        self.updateMainWindowTitle(self.state.baseName)
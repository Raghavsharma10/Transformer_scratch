def retrain(self, name, ids, labels):
        """Reentrenar parcialmente un clasificador SVM.

        Args:
            name (str): Nombre para el clasidicador.
            ids (list): Se espera una lista de N ids de textos ya almacenados
                en el TextClassifier.
            labels (list): Se espera una lista de N etiquetas. Una por cada id
                de texto presente en ids.
        Nota:
            Usa el clasificador de `Scikit-learn <http://scikit-learn.org/>`_
        """
        if not all(np.in1d(ids, self.ids)):
            raise ValueError("Hay ids de textos que no se encuentran \
                              almacenados.")
        try:
            classifier = getattr(self, name)
        except AttributeError:
            raise AttributeError("No hay ningun clasificador con ese nombre.")
        indices = np.in1d(self.ids, ids)
        if isinstance(labels, str):
            labels = [labels]
        classifier.partial_fit(self.tfidf_mat[indices, :], labels)
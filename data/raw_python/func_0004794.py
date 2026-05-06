def make_classifier(self, name, ids, labels):
        """Entrenar un clasificador SVM sobre los textos cargados.

        Crea un clasificador que se guarda en el objeto bajo el nombre `name`.

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
        setattr(self, name, SGDClassifier())
        classifier = getattr(self, name)
        indices = np.searchsorted(self.ids, ids)
        classifier.fit(self.tfidf_mat[indices, :], labels)
def reload_texts(self, texts, ids, vocabulary=None):
        """Calcula los vectores de terminos de textos y los almacena.

        A diferencia de :func:`~TextClassifier.TextClassifier.store_text` esta
        funcion borra cualquier informacion almacenada y comienza el conteo
        desde cero. Se usa para redefinir el vocabulario sobre el que se
        construyen los vectores.

        Args:
            texts (list): Una lista de N textos a incorporar.
            ids (list): Una lista de N ids alfanumericos para los textos.
        """
        self._check_id_length(ids)
        self.ids = np.array(sorted(ids))
        if vocabulary:
            self.vectorizer.vocabulary = vocabulary
        sorted_texts = [x for (y, x) in sorted(zip(ids, texts))]
        self.term_mat = self.vectorizer.fit_transform(sorted_texts)
        self._update_tfidf()
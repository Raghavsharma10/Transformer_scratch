def classify(self, classifier_name, examples, max_labels=None,
                 goodness_of_fit=False):
        """Usar un clasificador SVM para etiquetar textos nuevos.

        Args:
            classifier_name (str): Nombre del clasidicador a usar.
            examples (list or str): Se espera un ejemplo o una lista de
                ejemplos a clasificar en texto plano o en ids.
            max_labels (int, optional): Cantidad de etiquetas a devolver para
                cada ejemplo. Si se devuelve mas de una el orden corresponde a
                la plausibilidad de cada etiqueta. Si es None devuelve todas
                las etiquetas posibles.
            goodness_of_fit (bool, optional): Indica si devuelve o no una
                medida de cuan buenas son las etiquetas.
        Nota:
            Usa el clasificador de `Scikit-learn <http://scikit-learn.org/>`_

        Returns:
            tuple (array, array): (labels_considerados, puntajes)
                labels_considerados: Las etiquetas que se consideraron para
                    clasificar.
                puntajes: Cuanto más alto el puntaje, más probable es que la
                    etiqueta considerada sea la adecuada.
        """
        classifier = getattr(self, classifier_name)
        texts_vectors = self._make_text_vectors(examples)
        return classifier.classes_, classifier.decision_function(texts_vectors)
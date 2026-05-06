def _calculate_dispersion(X: Union[pd.DataFrame, np.ndarray], labels: np.ndarray, centroids: np.ndarray) -> float:
        """
        Calculate the dispersion between actual points and their assigned centroids
        """
        disp = np.sum(np.sum([np.abs(inst - centroids[label]) ** 2 for inst, label in zip(X, labels)]))  # type: float
        return disp
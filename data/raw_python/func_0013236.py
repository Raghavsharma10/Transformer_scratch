def plot_history(self, tid, scores=["loss", "f1", "accuracy"],
                     figsize=(15, 3)):
        """Plot the loss curves"""
        history = self.train_history(tid)
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=figsize)
        for i, score in enumerate(scores):
            plt.subplot(1, len(scores), i + 1)
            plt.tight_layout()
            plt.plot(history[score], label="train")
            plt.plot(history['val_' + score], label="validation")
            plt.title(score)
            plt.ylabel(score)
            plt.xlabel('epoch')
            plt.legend(loc='best')
        return fig
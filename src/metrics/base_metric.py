from abc import abstractmethod


class BaseMetric:
    """
    Base class for all metrics
    """

    def __init__(self, text_encoder, name=None, *args, **kwargs):
        """
        Args:
            name (str | None): metric name to use in logger and writer.
        """
        self.name = name if name is not None else type(self).__name__
        self.text_encoder = text_encoder

    @abstractmethod
    def __call__(self, **batch):
        """
        Defines metric calculation logic for a given batch.
        Can use external functions (like TorchMetrics) or custom ones.
        """
        raise NotImplementedError()

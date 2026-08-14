import numpy as np

class MinMaxNormalizer:
    """
    Min-Max normalizer that fits min and max bounds on a set of features (gallery)
    and applies the transformation consistently to gallery and query features.
    """
    def __init__(self, min_val=None, max_val=None):
        self.min_val = np.array(min_val) if min_val is not None else None
        self.max_val = np.array(max_val) if max_val is not None else None
        
    def fit(self, features):
        """
        Fits the normalizer by finding the column-wise minimum and maximum values.
        
        Args:
            features (np.ndarray): 2D array of shape (N, D).
        """
        self.min_val = np.min(features, axis=0)
        self.max_val = np.max(features, axis=0)
        
    def transform(self, features):
        """
        Applies the fitted normalization to the features.
        
        Args:
            features (np.ndarray): 2D array of shape (N, D).
            
        Returns:
            np.ndarray: Normalized features of shape (N, D) in range [0, 1].
        """
        if self.min_val is None or self.max_val is None:
            raise ValueError("Normalizer must be fitted before transformation.")
            
        range_val = self.max_val - self.min_val
        # If min and max are identical (range is 0), prevent division by zero by setting range to 1.0.
        range_val = np.where(range_val < 1e-8, 1.0, range_val)
        
        return (features - self.min_val) / range_val
        
    def fit_transform(self, features):
        self.fit(features)
        return self.transform(features)

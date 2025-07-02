import os
import joblib
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd


class HHatPredictor:
    """Encapsulates a pre-trained **H-hat** (Ĥ) regression / classification model.

    The class is intentionally lightweight so it can be embedded in batch or real-time
    inference pipelines.  A lazily-loaded scikit-learn object is internally used to
    compute predictions.
    """

    _DEFAULT_MODEL_FILENAME = "h_hat_predictor_model_sk152.pkl"

    def __init__(self, model_path: Union[str, os.PathLike, None] = None) -> None:
        """Instantiate the predictor and immediately load the serialized model.

        Parameters
        ----------
        model_path:
            Optional path to the *\.pkl* file containing a scikit-learn compatible
            estimator / pipeline.  If *None*, a file named
            ``h_hat_predictor_model_sk152.pkl`` is searched relative to this module's
            directory and then relative to the current working directory.
        """
        self._model_path: str = self._resolve_model_path(model_path)
        self._model: Any = self._load_model_from_disk(self._model_path)

    # ------------------------------------------------------------------
    # public helpers
    # ------------------------------------------------------------------

    def predict_from_params(self, params: Dict[str, Union[int, float]]) -> Union[float, np.ndarray]:
        """Return the model prediction for a single parameter dictionary.

        The input *params* **must** map feature names (as expected by the trained
        model) to numeric values.  A *pandas.DataFrame* is constructed internally
        so that the feature ordering matches the training set:

        1.  If the model exposes the *feature_names_in_* attribute (available on
            scikit-learn >= 1.0), the input is re-ordered accordingly.
        2.  Otherwise we rely on the order provided by *params* – in this case it
            is the caller's responsibility to use the correct ordering.
        """
        if not isinstance(params, dict):
            raise TypeError("params must be a dictionary mapping feature names to values")

        X = self._prepare_dataframe(params)
        preds = self._model.predict(X)
        # ``predict`` might return shape (1,) or scalar depending on the model –
        # normalise the value we return.
        return preds[0] if preds.shape == (1,) else preds

    def get_feature_importances(self) -> List[float]:
        """Return the model-reported feature importances, if available.

        For tree ensembles the attribute is *feature_importances_*, while for
        linear models we expose the absolute value of the *coef_* attribute.
        When neither attribute is present an *AttributeError* is raised.
        """
        if hasattr(self._model, "feature_importances_"):
            return self._model.feature_importances_.tolist()
        if hasattr(self._model, "coef_"):
            coefs = getattr(self._model, "coef_")
            # flatten and take absolute value for better interpretability
            return np.abs(coefs).ravel().tolist()
        raise AttributeError("Underlying model does not expose feature importances or coefficients")

    # ------------------------------------------------------------------
    # internal implementation details
    # ------------------------------------------------------------------

    @staticmethod
    def _load_model_from_disk(path: str) -> Any:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        return joblib.load(path)

    @staticmethod
    def _resolve_model_path(model_path: Union[str, os.PathLike, None]) -> str:
        if model_path is not None:
            return os.fspath(model_path)

        # attempt 1 – alongside this python file
        here_dir = os.path.dirname(__file__)
        candidate = os.path.join(here_dir, HHatPredictor._DEFAULT_MODEL_FILENAME)
        if os.path.isfile(candidate):
            return candidate

        # attempt 2 – current working directory (useful when packaged separately)
        candidate = os.path.join(os.getcwd(), HHatPredictor._DEFAULT_MODEL_FILENAME)
        if os.path.isfile(candidate):
            return candidate

        raise FileNotFoundError(
            "Unable to resolve default model path.  Please supply `model_path` explicitly."
        )

    def _prepare_dataframe(self, params: Dict[str, Union[int, float]]) -> pd.DataFrame:
        # If the model offers feature ordering info we respect it
        feature_order: List[str]
        if hasattr(self._model, "feature_names_in_"):
            feature_order = list(getattr(self._model, "feature_names_in_"))
        else:
            feature_order = list(params.keys())

        # Construct DataFrame with a single row in the correct order
        data = {feature: params.get(feature, np.nan) for feature in feature_order}
        df = pd.DataFrame([data], columns=feature_order)

        # Sanity check – if any NaNs remain the user did not supply all required features
        if df.isna().any(axis=None):
            missing = df.columns[df.isna().any()].tolist()
            raise ValueError(f"Missing required feature values for: {missing}")

        return df
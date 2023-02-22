from copy import deepcopy
from openml.tasks.task import TaskType
from openml.extensions.sklearn import SklearnExtension
from openml.flows import OpenMLFlow
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, cast, Sized
from skactiveml.base import SingleAnnotatorPoolQueryStrategy, SkactivemlClassifier


import inspect
import json
import logging
from collections import OrderedDict  # noqa: F401
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, cast, Sized
import warnings

import numpy as np
import pandas as pd
import scipy.stats
import scipy.sparse
import sklearn.base
import sklearn.model_selection
import sklearn.pipeline

import openml
from openml.exceptions import PyOpenMLError
from openml.flows import OpenMLFlow
from openml.runs.trace import OpenMLRunTrace
from openml.tasks import (
    OpenMLTask,
    OpenMLSupervisedTask,
    OpenMLClusteringTask,
    OpenMLRegressionTask,
    OpenMLActiveClassificationTask
)

logger = logging.getLogger(__name__)

class SkactivemlExtension(SklearnExtension):
    @classmethod
    def can_handle_flow(cls, flow: "OpenMLFlow") -> bool:
        """Check whether a given describes a scikit-learn estimator.

        This is done by parsing the ``external_version`` field.

        Parameters
        ----------
        flow : OpenMLFlow

        Returns
        -------
        bool
        """
        return cls._is_skactiveml_flow(flow)

    @classmethod
    def _is_skactiveml_flow(cls, flow: OpenMLFlow) -> bool:
        if getattr(flow, "dependencies", None) is not None and "skactiveml" in flow.dependencies:
            return True
        if flow.external_version is None:
            return False
        else:
            return (
                flow.external_version.startswith("skactiveml==")
                or ",skactiveml==" in flow.external_version
            )

    @classmethod
    def can_handle_model(cls, model: Any) -> bool:
        """Check whether a model is an instance of ``sklearn.base.BaseEstimator``.

        Parameters
        ----------
        model : Any

        Returns
        -------
        bool
        """
        if isinstance(model, dict) and len(model) == 4:
            return (
                isinstance(model.get('query_strategy', None), SingleAnnotatorPoolQueryStrategy)
                and isinstance(model.get('prediction_model', None), SkactivemlClassifier)
                and isinstance(model.get('query_params', None), (dict, type(None)))
                and isinstance(model.get('budget', None), (int, float, type(None)))
            )
        return False

    def model_to_flow(self, model: Any) -> "OpenMLFlow":
        """Transform a scikit-learn model to a flow for uploading it to OpenML.

        Parameters
        ----------
        model : Any

        Returns
        -------
        OpenMLFlow
        """
        # Necessary to make pypy not complain about all the different possible return types

        # flow = self._serialize_sklearn(model['prediction_model'])
        flow = self._serialize_sklearn(model['query_strategy'])

        flow.model = OrderedDict()

        flow.model['query_strategy'] = model['query_strategy']
        flow.model['prediction_model'] = model['prediction_model']
        flow.model['query_params'] = model['query_params']
        flow.model['budget'] = model['budget']
        return flow

    def is_estimator(self, model: Any) -> bool:
        """Check whether the given model is a scikit-learn estimator.

        This function is only required for backwards compatibility and will be removed in the
        near future.

        Parameters
        ----------
        model : Any

        Returns
        -------
        bool
        """
        o = model
        return (hasattr(o, "fit") or hasattr(o, "query")) and hasattr(o, "get_params") and hasattr(o, "set_params")

    def check_if_model_fitted(self, model: Any) -> bool:
        """Returns True/False denoting if the model has already been fitted/trained

        Parameters
        ----------
        model : Any

        Returns
        -------
        bool
        """
        # try:
        #     # check if model is fitted
        #     from sklearn.exceptions import NotFittedError

        #     # Creating random dummy data of arbitrary size
        #     dummy_data = np.random.uniform(size=(10, 3))
        #     # Using 'predict' instead of 'sklearn.utils.validation.check_is_fitted' for a more
        #     # robust check that works across sklearn versions and models. Internally, 'predict'
        #     # should call 'check_is_fitted' for every concerned attribute, thus offering a more
        #     # assured check than explicit calls to 'check_is_fitted'
        #     model.predict(dummy_data)
        #     # Will reach here if the model was fit on a dataset with 3 features
        #     return True
        # except NotFittedError:  # needs to be the first exception to be caught
        #     # Model is not fitted, as is required
        #     return False
        # except ValueError:
        #     # Will reach here if the model was fit on a dataset with more or less than 3 features
        #     return True
        return False

    def _extract_sklearn_parameter_docstring(self, model) -> Union[None, str]:
        """Extracts the part of sklearn docstring containing parameter information

        Fetches the entire docstring and trims just the Parameter section.
        The assumption is that 'Parameters' is the first section in sklearn docstrings,
        followed by other sections titled 'Attributes', 'See also', 'Note', 'References',
        appearing in that order if defined.
        Returns a None if no section with 'Parameters' can be found in the docstring.

        Parameters
        ----------
        model : sklearn model

        Returns
        -------
        str, or None
        """

        def match_format(s):
            return "{}\n{}\n".format(s, len(s) * "-")

        s = inspect.getdoc(model)
        if s is None:
            return None
        try:
            index1 = s.index(match_format("Parameters"))
        except ValueError as e:
            # when sklearn docstring has no 'Parameters' section
            logger.warning("{} {}".format(match_format("Parameters"), e))
            return None

        # headings = ["Attributes", "Notes", "See also", "Note", "References"]
        # for h in headings:
        #     try:
        #         # to find end of Parameters section
        #         index2 = s.index(match_format(h))
        #         break
        #     except ValueError:
        #         logger.warning("{} not available in docstring".format(h))
        #         continue
        # else:
        #     # in the case only 'Parameters' exist, trim till end of docstring
        #     index2 = len(s)
        index2 = len(s)
        s = s[index1:index2]
        return s.strip()


    def _get_sklearn_description(self, model: Any, char_lim: int = 1024) -> str:
        """Fetches the sklearn function docstring for the flow description

        Retrieves the sklearn docstring available and does the following:
        * If length of docstring <= char_lim, then returns the complete docstring
        * Else, trims the docstring till it encounters a 'Read more in the :ref:'
        * Or till it encounters a 'Parameters\n----------\n'
        The final string returned is at most of length char_lim with leading and
        trailing whitespaces removed.

        Parameters
        ----------
        model : sklearn model
        char_lim : int
            Specifying the max length of the returned string.
            OpenML servers have a constraint of 1024 characters for the 'description' field.

        Returns
        -------
        str
        """

        def match_format(s):
            return "{}\n{}\n".format(s, len(s) * "-")

        s = inspect.getdoc(model)
        if s is None:
            return ""
        # try:
        #     # trim till 'Read more'
        #     pattern = "Read more in the :ref:"
        #     index = s.index(pattern)
        #     s = s[:index]
        #     # trimming docstring to be within char_lim
        #     if len(s) > char_lim:
        #         s = "{}...".format(s[: char_lim - 3])
        #     return s.strip()
        # except ValueError:
        #     logger.warning(
        #         "'Read more' not found in descriptions. "
        #         "Trying to trim till 'Parameters' if available in docstring."
        #     )
        #     pass
        try:
            # if 'Read more' doesn't exist, trim till 'Parameters'
            pattern = "Parameters"
            index = s.index(match_format(pattern))
        except ValueError:
            # returning full docstring
            logger.warning("'Parameters' not found in docstring. Omitting docstring trimming.")
            index = len(s)
        s = s[:index]
        # trimming docstring to be within char_lim
        if len(s) > char_lim:
            s = "{}...".format(s[: char_lim - 3])
        return s.strip()


    def seed_model(self, model: Any, seed: Optional[int] = None) -> Any:
        """Set the random state of all the unseeded components of a model and return the seeded
        model.

        Required so that all seed information can be uploaded to OpenML for reproducible results.

        Models that are already seeded will maintain the seed. In this case,
        only integer seeds are allowed (An exception is raised when a RandomState was used as
        seed).

        Parameters
        ----------
        model : sklearn model
            The model to be seeded
        seed : int
            The seed to initialize the RandomState with. Unseeded subcomponents
            will be seeded with a random number from the RandomState.

        Returns
        -------
        Any
        """

        for k, v in model.items():
            if k != 'query_params' and hasattr(v, "get_params") and hasattr(v, "set_params"):
                model[k].model = super().seed_model(v, seed)

        for k, v in model['query_params'].items():
            if hasattr(v, "get_params") and hasattr(v, "set_params"):
                model['query_params'][k] = super().seed_model(v, seed)

        return model

    def _run_model_on_fold(
        self,
        model: Any,
        task: "OpenMLTask",
        X_train: Union[np.ndarray, scipy.sparse.spmatrix, pd.DataFrame],
        rep_no: int,
        fold_no: int,
        y_train: Optional[np.ndarray] = None,
        X_test: Optional[Union[np.ndarray, scipy.sparse.spmatrix, pd.DataFrame]] = None,
    ) -> Tuple[
        np.ndarray, Optional[pd.DataFrame], "OrderedDict[str, float]", Optional[OpenMLRunTrace]
    ]:
        """Run a model on a repeat,fold,subsample triplet of the task and return prediction
        information.

        Furthermore, it will measure run time measures in case multi-core behaviour allows this.
        * exact user cpu time will be measured if the number of cores is set (recursive throughout
        the model) exactly to 1
        * wall clock time will be measured if the number of cores is set (recursive throughout the
        model) to any given number (but not when it is set to -1)

        Returns the data that is necessary to construct the OpenML Run object. Is used by
        run_task_get_arff_content. Do not use this function unless you know what you are doing.

        Parameters
        ----------
        model : Any
            The UNTRAINED model to run. The model instance will be copied and not altered.
        task : OpenMLTask
            The task to run the model on.
        X_train : array-like
            Training data for the given repetition and fold.
        rep_no : int
            The repeat of the experiment (0-based; in case of 1 time CV, always 0)
        fold_no : int
            The fold nr of the experiment (0-based; in case of holdout, always 0)
        y_train : Optional[np.ndarray] (default=None)
            Target attributes for supervised tasks. In case of classification, these are integer
            indices to the potential classes specified by dataset.
        X_test : Optional, array-like (default=None)
            Test attributes to test for generalization in supervised tasks.

        Returns
        -------
        pred_y : np.ndarray
            Predictions on the training/test set, depending on the task type.
            For supervised tasks, predictions are on the test set.
            For unsupervised tasks, predictions are on the training set.
        proba_y : pd.DataFrame, optional
            Predicted probabilities for the test set.
            None, if task is not Classification or Learning Curve prediction.
        user_defined_measures : OrderedDict[str, float]
            User defined measures that were generated on this fold
        trace : OpenMLRunTrace, optional
            arff trace object from a fitted model and the trace content obtained by
            repeatedly calling ``run_model_on_task``
        """

        def _prediction_to_probabilities(
            y: Union[np.ndarray, List], model_classes: List[Any], class_labels: Optional[List[str]]
        ) -> pd.DataFrame:
            """Transforms predicted probabilities to match with OpenML class indices.

            Parameters
            ----------
            y : np.ndarray
                Predicted probabilities (possibly omitting classes if they were not present in the
                training data).
            model_classes : list
                List of classes known_predicted by the model, ordered by their index.
            class_labels : list
                List of classes as stored in the task object fetched from server.

            Returns
            -------
            pd.DataFrame
            """
            if class_labels is None:
                raise ValueError("The task has no class labels")

            if isinstance(y_train, np.ndarray) and isinstance(class_labels[0], str):
                # mapping (decoding) the predictions to the categories
                # creating a separate copy to not change the expected pred_y type
                y = [class_labels[pred] for pred in y]  # list or numpy array of predictions

            # model_classes: sklearn classifier mapping from original array id to
            # prediction index id
            if not isinstance(model_classes, list):
                raise ValueError("please convert model classes to list prior to calling this fn")

            # DataFrame allows more accurate mapping of classes as column names
            result = pd.DataFrame(
                0, index=np.arange(len(y)), columns=model_classes, dtype=np.float32
            )
            for obs, prediction in enumerate(y):
                result.loc[obs, prediction] = 1.0
            return result

        query_strategy = sklearn.base.clone(model['query_strategy'])
        prediction_model = sklearn.base.clone(model['prediction_model'])
        query_params = deepcopy(model['query_params'])

        if isinstance(task, OpenMLSupervisedTask):
            if y_train is None:
                raise TypeError("argument y_train must not be of type None")
            if X_test is None:
                raise TypeError("argument X_test must not be of type None")

        # sanity check: prohibit users from optimizing n_jobs
        self._prevent_optimize_n_jobs(prediction_model)
        for _, v in query_params.items():
            self._prevent_optimize_n_jobs(v)

        self._prevent_optimize_n_jobs(query_params)
        # measures and stores runtimes
        user_defined_measures = OrderedDict()  # type: 'OrderedDict[str, float]'
        try:
            # for measuring runtime. Only available since Python 3.3
            modelfit_start_cputime = time.process_time()
            modelfit_start_walltime = time.time()

            queries_t = []
            pred_t = []
            proba_t = []
            budget_t = []

            if isinstance(task, OpenMLActiveClassificationTask):
                y = np.full(shape=len(y_train), fill_value=None)
                # for c in range(model.budget):
                max_budget = model['budget']
                if max_budget is None:
                    max_budget = np.inf
                cycle = 0
                used_budget=0
                annotation_cost = np.full(len(y), 1)
                if task.annotation_costs is not None:
                    annotation_cost = task.annotation_costs
                max_cycles = len(y)-1
                while used_budget < max_budget and cycle < max_cycles:
                    query_idx = query_strategy.query(X=X_train, y=y, **query_params)
                    y[query_idx] = y_train.values[query_idx]
                    used_budget = used_budget + annotation_cost[query_idx].item()

                    budget_t.append(used_budget)
                    queries_t.append(query_idx)

                    prediction_model.fit(X_train, y)
                    pred_t.append(prediction_model.predict(X_test))
                    proba_t_raw = prediction_model.predict_proba(X_test)
                    proba_t.append(pd.DataFrame(proba_t_raw, columns=prediction_model.classes_))
                    cycle += 1

                prediction_model.fit(X_train, y)

            modelfit_dur_cputime = (time.process_time() - modelfit_start_cputime) * 1000
            modelfit_dur_walltime = (time.time() - modelfit_start_walltime) * 1000

            user_defined_measures['budget_t'] = np.array(budget_t)
            user_defined_measures['queries_t'] = np.array(queries_t)
            user_defined_measures['pred_t'] = np.array(pred_t)
            user_defined_measures['proba_t'] = proba_t

            user_defined_measures["usercpu_time_millis_training"] = modelfit_dur_cputime
            refit_time = prediction_model.refit_time_ * 1000 if hasattr(prediction_model, "refit_time_") else 0
            user_defined_measures["wall_clock_time_millis_training"] = modelfit_dur_walltime

        except AttributeError as e:
            # typically happens when training a regressor on classification task
            raise PyOpenMLError(str(e))

        if isinstance(task, OpenMLActiveClassificationTask):
            # search for model classes_ (might differ depending on modeltype)
            # first, pipelines are a special case (these don't have a classes_
            # object, but rather borrows it from the last step. We do this manually,
            # because of the BaseSearch check)
            if isinstance(prediction_model, sklearn.pipeline.Pipeline):
                used_estimator = prediction_model.steps[-1][-1]
            else:
                used_estimator = prediction_model

            if self._is_hpo_class(used_estimator):
                model_classes = used_estimator.best_estimator_.classes_
            else:
                model_classes = used_estimator.classes_

            if not isinstance(model_classes, list):
                model_classes = model_classes.tolist()

            # to handle the case when dataset is numpy and categories are encoded
            # however the class labels stored in task are still categories
            if isinstance(y_train, np.ndarray) and isinstance(
                cast(List, task.class_labels)[0], str
            ):
                model_classes = [cast(List[str], task.class_labels)[i] for i in model_classes]

        modelpredict_start_cputime = time.process_time()
        modelpredict_start_walltime = time.time()

        # In supervised learning this returns the predictions for Y, in clustering
        # it returns the clusters
        if isinstance(task, OpenMLActiveClassificationTask):
            pred_y = prediction_model.predict(X_test)
        else:
            raise ValueError(task)

        modelpredict_duration_cputime = (time.process_time() - modelpredict_start_cputime) * 1000
        user_defined_measures["usercpu_time_millis_testing"] = modelpredict_duration_cputime
        user_defined_measures["usercpu_time_millis"] = (
            modelfit_dur_cputime + modelpredict_duration_cputime
        )
        modelpredict_duration_walltime = (time.time() - modelpredict_start_walltime) * 1000
        user_defined_measures["wall_clock_time_millis_testing"] = modelpredict_duration_walltime
        user_defined_measures["wall_clock_time_millis"] = (
            modelfit_dur_walltime + modelpredict_duration_walltime + refit_time
        )

        if isinstance(task, OpenMLActiveClassificationTask):

            try:
                proba_y = prediction_model.predict_proba(X_test)
                proba_y = pd.DataFrame(proba_y, columns=model_classes)  # handles X_test as numpy
            except AttributeError:  # predict_proba is not available when probability=False
                proba_y = _prediction_to_probabilities(pred_y, model_classes, task.class_labels)

            if task.class_labels is not None:
                if proba_y.shape[1] != len(task.class_labels):
                    # Remap the probabilities in case there was a class missing
                    # at training time. By default, the classification targets
                    # are mapped to be zero-based indices to the actual classes.
                    # Therefore, the model_classes contain the correct indices to
                    # the correct probability array. Example:
                    # classes in the dataset: 0, 1, 2, 3, 4, 5
                    # classes in the training set: 0, 1, 2, 4, 5
                    # then we need to add a column full of zeros into the probabilities
                    # for class 3 because the rest of the library expects that the
                    # probabilities are ordered the same way as the classes are ordered).
                    message = "Estimator only predicted for {}/{} classes!".format(
                        proba_y.shape[1],
                        len(task.class_labels),
                    )
                    warnings.warn(message)
                    openml.config.logger.warning(message)

                    for i, col in enumerate(task.class_labels):
                        # adding missing columns with 0 probability
                        if col not in model_classes:
                            proba_y[col] = 0
                    # We re-order the columns to move possibly added missing columns into place.
                    proba_y = proba_y[task.class_labels]
            else:
                raise ValueError("The task has no class labels")

            if not np.all(set(proba_y.columns) == set(task.class_labels)):
                missing_cols = list(set(task.class_labels) - set(proba_y.columns))
                raise ValueError("Predicted probabilities missing for the columns: ", missing_cols)

        elif isinstance(task, OpenMLRegressionTask):
            proba_y = None

        elif isinstance(task, OpenMLClusteringTask):
            proba_y = None

        else:
            raise TypeError(type(task))

        if self._is_hpo_class(prediction_model):
            trace_data = self._extract_trace_data(prediction_model, rep_no, fold_no)
            trace = self._obtain_arff_trace(
                prediction_model, trace_data
            )  # type: Optional[OpenMLRunTrace]  # noqa E501
        else:
            trace = None

        return pred_y, proba_y, user_defined_measures, trace

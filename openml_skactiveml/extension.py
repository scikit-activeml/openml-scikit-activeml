from copy import deepcopy
from timeit import repeat
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

from sklearn.base import  BaseEstimator
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

from collections.abc import Iterable

import sys
if sys.version_info >= (3, 5):
    from json.decoder import JSONDecodeError
else:
    JSONDecodeError = ValueError

SKLEARN_PIPELINE_STRING_COMPONENTS = ("drop", "passthrough")
COMPONENT_REFERENCE = "component_reference"
COMPOSITION_STEP_CONSTANT = "composition_step_constant"

logger = logging.getLogger(__name__)


def check_skactiveml_params(object, param_name, param_value):
    if not isinstance(object, (SkactivemlClassifier, SingleAnnotatorPoolQueryStrategy)):
        return
    attributes = dir(object)
    for attr_name in attributes:
        if hasattr(object, attr_name):
            value = getattr(object, attr_name)
            if attr_name == param_name:
                if getattr(object, param_name) is not None:
                    raise ValueError(f'An object of type `{type(object)}` has `{param_name}` that are not set to `None`.')
                setattr(object, param_name, param_value)
            elif isinstance(value, (SkactivemlClassifier, SingleAnnotatorPoolQueryStrategy)):
                check_skactiveml_params(value, param_name, param_value)
            elif isinstance(value, dict):
                for item in value.values():
                    check_skactiveml_params(item, param_name, param_value)
            elif isinstance(value, list):
                for item in value:
                    check_skactiveml_params(item, param_name, param_value)


class PoolSkactivemlModel:
    """Pool Skactiveml Model

    This class implements the model that is used for an active learning 
    experiment. It consists of the query_strategy, a prediction model,
    a selection model (if applicable), extra query parameters and the
    budget.

    Parameters
    ----------
    query_strategy : skactiveml.base.SingleAnnotatorPoolQueryStrategy
        The query strategy used for the instance selection.
    prediction_model : skactiveml.base.SkactivemlClassifier
        The classifier used for the evaluation.
    selection_model : skactiveml.base.SkactivemlClassifier, optional (default=None)
        The classifier used for the instance selection. If selection_model and selection_model_name are None, no extra parameters will be passed.
    selection_model_name : str, optional (default=None)
        The selection model name within `query_strategy.query`. If selection_model and selection_model_name are None, no extra parameters will be passed.
    extra_query_params : dict-like, optional (default=None)
        Extra parameters for `query_strategy.query`. If None, no extra parameters will be passed.
    budget : int, default=-1
        The maximum number of labeled instances to query. If the number is -1, all instances will be queried.
    """
    def __init__(self, query_strategy, prediction_model, selection_model=None, selection_model_name=None, extra_query_params=None, budget=-1):
        self.query_strategy = query_strategy
        self.prediction_model = prediction_model
        self.selection_model = selection_model
        self.selection_model_name = selection_model_name
        self.extra_query_params = extra_query_params
        self.budget = budget

        if (self.selection_model is None) != (self.selection_model_name is None):
            raise ValueError(f"if `selection_model` or `selection_model_name` is None, the other has to be `None`, too." )
        
        estimators_to_check = [query_strategy, prediction_model, selection_model]
        if extra_query_params:
            estimators_to_check += list(extra_query_params.values())

        for estimator in estimators_to_check:
            check_skactiveml_params(estimator, 'classes', None)
            check_skactiveml_params(estimator, 'missing_label', None)
                

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
        return False

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
        return isinstance(model, PoolSkactivemlModel)

    def model_to_flow(self, model: Any) -> "OpenMLFlow":
        """Transform a scikit-learn model to a flow for uploading it to OpenML.

        Parameters
        ----------
        model : Any

        Returns
        -------
        OpenMLFlow
        """
        if isinstance(model, PoolSkactivemlModel):
            model = SkactivemlExtension._wrap_skactiveml_model(model)
        return self._serialize_sklearn(model)

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
        if isinstance(model, PoolSkactivemlModel):
            model = SkactivemlExtension._wrap_skactiveml_model(model)
        return hasattr(model, "get_params") and hasattr(model, "set_params")

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

        if isinstance(model, PoolSkactivemlModel):
            model = SkactivemlExtension._wrap_skactiveml_model(model)
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

    def _get_dependencies(self) -> str:
        import re
        dependencies = self._min_dependency_str(sklearn.__version__)
        # define the regular expression pattern
        # pattern = r"(scikit-learn|sklearn)\s*([>=<~!]*\d+[.\d]*)\n"

        # substitute the version number with "new_version"
        # dependencies = re.sub(pattern, "", dependencies)
        # print(dependencies)
        return dependencies

    def _get_external_version_string(
        self,
        model: Any,
        sub_components: Dict[str, OpenMLFlow],
    ) -> str:
        import re
        if isinstance(model, PoolSkactivemlModel):
            model = SkactivemlExtension._wrap_skactiveml_model(model)
        externel_extension = super(). _get_external_version_string(model=model, sub_components=sub_components)
        # define the regular expression pattern
        # pattern = r"(scikit-learn|sklearn)\s*([>=<~!]*\d+[.\d]*)"

        # substitute the version number with "new_version"
        # externel_extension = re.sub(pattern, "", externel_extension)
        # print(externel_extension)
        return externel_extension

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

        if isinstance(model, PoolSkactivemlModel):
            model = SkactivemlExtension._wrap_skactiveml_model(model)
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
        if isinstance(model, PoolSkactivemlModel):
            model = SkactivemlExtension._wrap_skactiveml_model(model)
        return super().seed_model(model=model, seed=seed)

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

        query_strategy = sklearn.base.clone(model.query_strategy)
        prediction_model = sklearn.base.clone(model.prediction_model)
        selection_model = deepcopy(model.selection_model)
        selection_model_name = deepcopy(model.selection_model_name)
        query_params = deepcopy(model.extra_query_params)
        if query_params is None:
            query_params = {}
        if selection_model is not None and selection_model_name is not None:
            query_params[selection_model_name] = selection_model

        if isinstance(task, OpenMLSupervisedTask):
            if y_train is None:
                raise TypeError("argument y_train must not be of type None")
            if X_test is None:
                raise TypeError("argument X_test must not be of type None")

        # replace classes
        # selection model is implicitely contained in query_params
        estimators_to_check = [query_strategy, prediction_model] + list(query_params.values())
        for estimator in estimators_to_check:
            check_skactiveml_params(estimator, 'classes', task.class_labels)

        # sanity check: prohibit users from optimizing n_jobs
        self._prevent_optimize_n_jobs(prediction_model)
        for _, v in query_params.items():
            self._prevent_optimize_n_jobs(v)

        # measures and stores runtimes
        user_defined_measures = OrderedDict()  # type: 'OrderedDict[str, float]'
        try:
            batch_size = task.batch_size
            if batch_size is None:
                batch_size=1
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
                max_budget = model.budget
                if max_budget == -1:
                    max_budget = np.inf
                cycle = 0
                used_budget=0
                annotation_costs = np.full(len(y), 1)
                if task.annotation_costs is not None:
                    train_idx = task.get_train_test_split_indices(fold=fold_no, repeat=rep_no)[0]
                    annotation_costs = task.annotation_costs[train_idx]
                max_cycles = (len(y)-1)/task.batch_size
                while used_budget < max_budget and cycle < max_cycles:
                    query_idxs, utilities = query_strategy.query(X=X_train, y=y, batch_size=batch_size, return_utilities=True, **query_params)
                    y[query_idxs] = y_train.values[query_idxs]
                    for query_idx in query_idxs:
                        used_budget = used_budget + annotation_costs[query_idx].item()

                    budget_t.append(used_budget)
                    queries_t.append(query_idxs.tolist())

                    prediction_model.fit(X_train, y)

                    if isinstance(prediction_model, sklearn.pipeline.Pipeline):
                        used_estimator = prediction_model.steps[-1][-1]
                    else:
                        used_estimator = prediction_model
                    if self._is_hpo_class(used_estimator):
                        model_classes = used_estimator.best_estimator_.classes_
                    else:
                        model_classes = used_estimator.classes_

                    pred_t.append(prediction_model.predict(X_test).tolist())
                    proba_t_raw = prediction_model.predict_proba(X_test)
                    proba_t_df = pd.DataFrame(proba_t_raw, columns=model_classes)
                    for _, col in enumerate(task.class_labels):
                        # adding missing columns with 0 probability
                        if col not in model_classes:
                            proba_t_df[col] = 0
                    proba_t_df = proba_t_df[task.class_labels]
                    proba_t.append(proba_t_df.values.tolist())

                    cycle += 1

                prediction_model.fit(X_train, y)

            modelfit_dur_cputime = (time.process_time() - modelfit_start_cputime) * 1000
            modelfit_dur_walltime = (time.time() - modelfit_start_walltime) * 1000

            user_defined_measures['budget_t'] = budget_t
            user_defined_measures['queries_t'] = queries_t
            user_defined_measures['pred_t'] = pred_t
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

            proba_y = prediction_model.predict_proba(X_test)
            proba_y = pd.DataFrame(proba_y, columns=model_classes)  # handles X_test as numpy

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

    def obtain_parameter_values(
        self,
        flow: "OpenMLFlow",
        model: Any = None,
    ) -> List[Dict[str, Any]]:
        if isinstance(model, PoolSkactivemlModel):
            model = SkactivemlExtension._wrap_skactiveml_model(model)
        parameters = super().obtain_parameter_values(flow=flow, model=model)
        return parameters


    @classmethod
    def _wrap_skactiveml_model(clf, model):
        class PoolSkactivemlModel(BaseEstimator):
            """Pool Skactiveml Model

            This class implements the model that is used for an active learning 
            experiment. It consists of the query_strategy, a prediction model,
            a selection model (if applicable), extra query parameters and the
            budget.

            Parameters
            ----------
            query_strategy : skactiveml.base.SingleAnnotatorPoolQueryStrategy
                The query strategy used for the instance selection.
            prediction_model : skactiveml.base.SkactivemlClassifier
                The classifier used for the evaluation.
            selection_model : skactiveml.base.SkactivemlClassifier, optional (default=None)
                The classifier used for the instance selection. If selection_model and selection_model_name are None, no extra parameters will be passed.
            selection_model_name : str, optional (default=None)
                The selection model name within `query_strategy.query`. If selection_model and selection_model_name are None, no extra parameters will be passed.
            extra_query_params : dict-like, optional (default=None)
                Extra parameters for `query_strategy.query`. If None, no extra parameters will be passed.
            budget : int, default=-1
                The maximum number of labeled instances to query. If the number is -1, all instances will be queried.
            """
            def __init__(self, query_strategy, prediction_model, selection_model, selection_model_name, extra_query_params, budget):
                self.query_strategy = query_strategy
                self.prediction_model = prediction_model
                self.selection_model = selection_model
                self.selection_model_name = selection_model_name
                self.extra_query_params = extra_query_params
                self.budget = budget

        return PoolSkactivemlModel(
            query_strategy=model.query_strategy,
            prediction_model=model.prediction_model,
            selection_model=model.selection_model,
            selection_model_name=model.selection_model_name,
            extra_query_params=model.extra_query_params,
            budget=model.budget,
        )


    def _serialize_model(self, model: Any) -> OpenMLFlow:
        """Create an OpenMLFlow.

        Calls `sklearn_to_flow` recursively to properly serialize the
        parameters to strings and the components (other models) to OpenMLFlows.

        Parameters
        ----------
        model : sklearn estimator

        Returns
        -------
        OpenMLFlow

        """

        # Get all necessary information about the model objects itself
        (
            parameters,
            parameters_meta_info,
            subcomponents,
            subcomponents_explicit,
        ) = self._extract_information_from_model(model)

        # Check that a component does not occur multiple times in a flow as this
        # is not supported by OpenML
        self._check_multiple_occurence_of_component_in_flow(model, subcomponents)

        # Create a flow name, which contains all components in brackets, e.g.:
        # RandomizedSearchCV(Pipeline(StandardScaler,AdaBoostClassifier(DecisionTreeClassifier)),
        # StandardScaler,AdaBoostClassifier(DecisionTreeClassifier))
        class_name = model.__module__ + "." + model.__class__.__name__

        # will be part of the name (in brackets)
        sub_components_names = ""
        for key in subcomponents:
            if isinstance(subcomponents[key], OpenMLFlow):
                name = subcomponents[key].name
            elif (
                isinstance(subcomponents[key], str)
                and subcomponents[key] in SKLEARN_PIPELINE_STRING_COMPONENTS
            ):
                name = subcomponents[key]
            else:
                raise TypeError(type(subcomponents[key]))
            if key in subcomponents_explicit:
                sub_components_names += "," + key + "=" + name
            else:
                sub_components_names += "," + name

        if sub_components_names:
            # slice operation on string in order to get rid of leading comma
            name = "%s(%s)" % (class_name, sub_components_names[1:])
        else:
            name = class_name
        short_name = SklearnExtension.trim_flow_name(name)

        # Get the external versions of all sub-components
        external_version = self._get_external_version_string(model, subcomponents)
        dependencies = self._get_dependencies()
        tags = self._get_tags()

        sklearn_description = self._get_sklearn_description(model)
        flow = OpenMLFlow(
            name=name,
            class_name=class_name,
            custom_name=short_name,
            description=sklearn_description,
            model=model,
            components=subcomponents,
            parameters=parameters,
            parameters_meta_info=parameters_meta_info,
            external_version=external_version,
            tags=tags,
            extension=self,
            language="English",
            dependencies=dependencies,
        )

        return flow


    def _extract_information_from_model(
        self,
        model: Any,
    ) -> Tuple[
        "OrderedDict[str, Optional[str]]",
        "OrderedDict[str, Optional[Dict]]",
        "OrderedDict[str, OpenMLFlow]",
        Set,
    ]:
        # This function contains four "global" states and is quite long and
        # complicated. If it gets to complicated to ensure it's correctness,
        # it would be best to make it a class with the four "global" states being
        # the class attributes and the if/elif/else in the for-loop calls to
        # separate class methods

        # stores all entities that should become subcomponents
        sub_components = OrderedDict()  # type: OrderedDict[str, OpenMLFlow]
        # stores the keys of all subcomponents that should become
        sub_components_explicit = set()
        parameters = OrderedDict()  # type: OrderedDict[str, Optional[str]]
        parameters_meta_info = OrderedDict()  # type: OrderedDict[str, Optional[Dict]]
        parameters_docs = self._extract_sklearn_param_info(model)

        model_parameters = model.get_params(deep=False)
        for k, v in sorted(model_parameters.items(), key=lambda t: t[0]):
            rval = self._serialize_sklearn(v, model)

            def flatten_all(list_):
                """Flattens arbitrary depth lists of lists (e.g. [[1,2],[3,[1]]] -> [1,2,3,1])."""
                for el in list_:
                    if isinstance(el, (list, tuple)) and len(el) > 0:
                        yield from flatten_all(el)
                    else:
                        yield el

            # In case rval is a list of lists (or tuples), we need to identify two situations:
            # - sklearn pipeline steps, feature union or base classifiers in voting classifier.
            #   They look like e.g. [("imputer", Imputer()), ("classifier", SVC())]
            # - a list of lists with simple types (e.g. int or str), such as for an OrdinalEncoder
            #   where all possible values for each feature are described: [[0,1,2], [1,2,5]]
            is_non_empty_list_of_lists_with_same_type = (
                isinstance(rval, (list, tuple))
                and len(rval) > 0
                and isinstance(rval[0], (list, tuple))
                and all([isinstance(rval_i, type(rval[0])) for rval_i in rval])
            )

            # Check that all list elements are of simple types.
            nested_list_of_simple_types = (
                is_non_empty_list_of_lists_with_same_type
                and all([isinstance(el, SIMPLE_TYPES) for el in flatten_all(rval)])
                and all(
                    [
                        len(rv) in (2, 3) and rv[1] not in SKLEARN_PIPELINE_STRING_COMPONENTS
                        for rv in rval
                    ]
                )
            )

            if is_non_empty_list_of_lists_with_same_type and not nested_list_of_simple_types:
                # If a list of lists is identified that include 'non-simple' types (e.g. objects),
                # we assume they are steps in a pipeline, feature union, or base classifiers in
                # a voting classifier.
                parameter_value = list()  # type: List
                reserved_keywords = set(model.get_params(deep=False).keys())

                for i, sub_component_tuple in enumerate(rval):
                    identifier = sub_component_tuple[0]
                    sub_component = sub_component_tuple[1]
                    sub_component_type = type(sub_component_tuple)
                    if not 2 <= len(sub_component_tuple) <= 3:
                        # length 2 is for {VotingClassifier.estimators,
                        # Pipeline.steps, FeatureUnion.transformer_list}
                        # length 3 is for ColumnTransformer
                        msg = "Length of tuple of type {} does not match assumptions".format(
                            sub_component_type
                        )
                        raise ValueError(msg)

                    if isinstance(sub_component, str):
                        if sub_component not in SKLEARN_PIPELINE_STRING_COMPONENTS:
                            msg = (
                                "Second item of tuple does not match assumptions. "
                                "If string, can be only 'drop' or 'passthrough' but"
                                "got %s" % sub_component
                            )
                            raise ValueError(msg)
                        else:
                            pass
                    elif isinstance(sub_component, type(None)):
                        msg = (
                            "Cannot serialize objects of None type. Please use a valid "
                            "placeholder for None. Note that empty sklearn estimators can be "
                            "replaced with 'drop' or 'passthrough'."
                        )
                        raise ValueError(msg)
                    elif not isinstance(sub_component, OpenMLFlow):
                        msg = (
                            "Second item of tuple does not match assumptions. "
                            "Expected OpenMLFlow, got %s" % type(sub_component)
                        )
                        raise TypeError(msg)

                    if identifier in reserved_keywords:
                        parent_model = "{}.{}".format(model.__module__, model.__class__.__name__)
                        msg = "Found element shadowing official " "parameter for %s: %s" % (
                            parent_model,
                            identifier,
                        )
                        raise PyOpenMLError(msg)

                    # when deserializing the parameter
                    sub_components_explicit.add(identifier)
                    if isinstance(sub_component, str):

                        external_version = self._get_external_version_string(None, {})
                        dependencies = self._get_dependencies()
                        tags = self._get_tags()

                        sub_components[identifier] = OpenMLFlow(
                            name=sub_component,
                            description="Placeholder flow for scikit-learn's string pipeline "
                            "members",
                            components=OrderedDict(),
                            parameters=OrderedDict(),
                            parameters_meta_info=OrderedDict(),
                            external_version=external_version,
                            tags=tags,
                            language="English",
                            dependencies=dependencies,
                            model=None,
                        )
                        component_reference = OrderedDict()  # type: Dict[str, Union[str, Dict]]
                        component_reference[
                            "oml-python:serialized_object"
                        ] = COMPOSITION_STEP_CONSTANT
                        cr_value = OrderedDict()  # type: Dict[str, Any]
                        cr_value["key"] = identifier
                        cr_value["step_name"] = identifier
                        if len(sub_component_tuple) == 3:
                            cr_value["argument_1"] = sub_component_tuple[2]
                        component_reference["value"] = cr_value
                    else:
                        sub_components[identifier] = sub_component
                        component_reference = OrderedDict()
                        component_reference["oml-python:serialized_object"] = COMPONENT_REFERENCE
                        cr_value = OrderedDict()
                        cr_value["key"] = identifier
                        cr_value["step_name"] = identifier
                        if len(sub_component_tuple) == 3:
                            cr_value["argument_1"] = sub_component_tuple[2]
                        component_reference["value"] = cr_value
                    parameter_value.append(component_reference)

                # Here (and in the elif and else branch below) are the only
                # places where we encode a value as json to make sure that all
                # parameter values still have the same type after
                # deserialization
                if isinstance(rval, tuple):
                    parameter_json = json.dumps(tuple(parameter_value))
                else:
                    parameter_json = json.dumps(parameter_value)
                parameters[k] = parameter_json

            elif isinstance(rval, OpenMLFlow):

                # A subcomponent, for example the base model in
                # AdaBoostClassifier
                sub_components[k] = rval
                sub_components_explicit.add(k)
                component_reference = OrderedDict()
                component_reference["oml-python:serialized_object"] = COMPONENT_REFERENCE
                cr_value = OrderedDict()
                cr_value["key"] = k
                cr_value["step_name"] = None
                component_reference["value"] = cr_value
                cr = self._serialize_sklearn(component_reference, model)
                parameters[k] = json.dumps(cr)

            # elif isinstance(rval, OrderedDict):
            #     o = OrderedDict()
            #     for dict_key, dict_value in rval.items():
            #         dict_key = self._serialize_sklearn(dict_key)
            #         dict_value = self._serialize_sklearn(dict_value)
            #         o[dict_key]=dict_value
            else:
                # a regular hyperparameter
                if not (hasattr(rval, "__len__") and len(rval) == 0):
                    rval = json.dumps(rval)
                    parameters[k] = rval
                else:
                    parameters[k] = None

            if parameters_docs is not None:
                data_type, description = parameters_docs[k]
                parameters_meta_info[k] = OrderedDict(
                    (("description", description), ("data_type", data_type))
                )
            else:
                parameters_meta_info[k] = OrderedDict((("description", None), ("data_type", None)))

        return parameters, parameters_meta_info, sub_components, sub_components_explicit
    
    
    def _check_multiple_occurence_of_component_in_flow(
        self,
        model: Any,
        sub_components: Dict[str, OpenMLFlow],
    ) -> None:
        # to_visit_stack = []  # type: List[OpenMLFlow]
        # to_visit_stack.extend(sub_components.values())
        # known_sub_components = set()  # type: Set[str]

        # while len(to_visit_stack) > 0:
        #     visitee = to_visit_stack.pop()
        #     if isinstance(visitee, str) and visitee in SKLEARN_PIPELINE_STRING_COMPONENTS:
        #         known_sub_components.add(visitee)
        #     elif visitee.name in known_sub_components:
        #         raise ValueError(
        #             "Found a second occurence of component %s when "
        #             "trying to serialize %s." % (visitee.name, model)
        #         )
        #     else:
        #         known_sub_components.add(visitee.name)
        #         to_visit_stack.extend(visitee.components.values())
        pass

    def _deserialize_sklearn(
        self,
        o: Any,
        components: Optional[Dict] = None,
        initialize_with_defaults: bool = False,
        recursion_depth: int = 0,
        strict_version: bool = True,
    ) -> Any:
        """Recursive function to deserialize a scikit-learn flow.

        This function inspects an object to deserialize and decides how to do so. This function
        delegates all work to the respective functions to deserialize special data structures etc.
        This function works on everything that has been serialized to OpenML: OpenMLFlow,
        components (which are flows themselves), functions, hyperparameter distributions (for
        random search) and the actual hyperparameter values themselves.

        Parameters
        ----------
        o : mixed
            the object to deserialize (can be flow object, or any serialized
            parameter value that is accepted by)

        components : Optional[dict]
            Components of the current flow being de-serialized. These will not be used when
            de-serializing the actual flow, but when de-serializing a component reference.

        initialize_with_defaults : bool, optional (default=False)
            If this flag is set, the hyperparameter values of flows will be
            ignored and a flow with its defaults is returned.

        recursion_depth : int
            The depth at which this flow is called, mostly for debugging
            purposes

        strict_version : bool, default=True
            Whether to fail if version requirements are not fulfilled.

        Returns
        -------
        mixed
        """

        logger.info(
            "-%s flow_to_sklearn START o=%s, components=%s, init_defaults=%s"
            % ("-" * recursion_depth, o, components, initialize_with_defaults)
        )
        depth_pp = recursion_depth + 1  # shortcut var, depth plus plus

        # First, we need to check whether the presented object is a json string.
        # JSON strings are used to encoder parameter values. By passing around
        # json strings for parameters, we make sure that we can flow_to_sklearn
        # the parameter values to the correct type.

        if isinstance(o, str):
            try:
                o = json.loads(o)
            except JSONDecodeError:
                pass

        if isinstance(o, dict):
            # Check if the dict encodes a 'special' object, which could not
            # easily converted into a string, but rather the information to
            # re-create the object were stored in a dictionary.
            if "oml-python:serialized_object" in o:
                serialized_type = o["oml-python:serialized_object"]
                value = o["value"]
                if serialized_type == "type":
                    rval = self._deserialize_type(value)
                elif serialized_type == "rv_frozen":
                    rval = self._deserialize_rv_frozen(value)
                elif serialized_type == "function":
                    rval = self._deserialize_function(value)
                elif serialized_type in (COMPOSITION_STEP_CONSTANT, COMPONENT_REFERENCE):
                    if serialized_type == COMPOSITION_STEP_CONSTANT:
                        pass
                    elif serialized_type == COMPONENT_REFERENCE:
                        value = self._deserialize_sklearn(
                            value, recursion_depth=depth_pp, strict_version=strict_version
                        )
                    else:
                        raise NotImplementedError(serialized_type)
                    assert components is not None  # Necessary for mypy
                    step_name = value["step_name"]
                    key = value["key"]
                    component = self._deserialize_sklearn(
                        components[key],
                        initialize_with_defaults=initialize_with_defaults,
                        recursion_depth=depth_pp,
                        strict_version=strict_version,
                    )
                    # The component is now added to where it should be used
                    # later. It should not be passed to the constructor of the
                    # main flow object.
                    del components[key]
                    if step_name is None:
                        rval = component
                    elif "argument_1" not in value:
                        rval = (step_name, component)
                    else:
                        rval = (step_name, component, value["argument_1"])
                elif serialized_type == "cv_object":
                    rval = self._deserialize_cross_validator(
                        value, recursion_depth=recursion_depth, strict_version=strict_version
                    )
                else:
                    raise ValueError("Cannot flow_to_sklearn %s" % serialized_type)

            else:
                rval = OrderedDict(
                    (
                        self._deserialize_sklearn(
                            o=key,
                            components=components,
                            initialize_with_defaults=initialize_with_defaults,
                            recursion_depth=depth_pp,
                            strict_version=strict_version,
                        ),
                        self._deserialize_sklearn(
                            o=value,
                            components=components,
                            initialize_with_defaults=initialize_with_defaults,
                            recursion_depth=depth_pp,
                            strict_version=strict_version,
                        ),
                    )
                    for key, value in sorted(o.items())
                )
        elif isinstance(o, (list, tuple)):
            rval = [
                self._deserialize_sklearn(
                    o=element,
                    components=components,
                    initialize_with_defaults=initialize_with_defaults,
                    recursion_depth=depth_pp,
                    strict_version=strict_version,
                )
                for element in o
            ]
            if isinstance(o, tuple):
                rval = tuple(rval)
        elif isinstance(o, (bool, int, float, str)) or o is None:
            rval = o
        elif isinstance(o, OpenMLFlow):
            if (not self._is_skactiveml_flow(o)) and (not SklearnExtension._is_sklearn_flow(o)):
                raise ValueError("Only sklearn flows can be reinstantiated")
            rval = self._deserialize_model(
                flow=o,
                keep_defaults=initialize_with_defaults,
                recursion_depth=recursion_depth,
                strict_version=strict_version,
            )
        else:
            raise TypeError(o)
        logger.info("-%s flow_to_sklearn END   o=%s, rval=%s" % ("-" * recursion_depth, o, rval))
        return rval
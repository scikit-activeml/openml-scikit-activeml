# scikit-activeml Extension for OpenML python

This is a work in progress and not compatible with [openml-python](https://github.com/openml/openml-python), yet.
If you are interested in testing it, use the following [openml-python fork](https://github.com/scikit-activeml/openml-python-OpenMLActiveClassificationTask) which supports OpenMLActiveClassificationTask and try it out on the openml test server (call `openml.config.start_using_configuration_for_example()` after importing openml).

## Open Issues to be Discussed with OpenML
- How to differ between extensions? Currently, scikit-learn checks only for the dependencies via `_is_sklearn_flow` 
in the file `openml.extensions.sklearn.extension.py`.
- We cannot store the results of the individual active learning cycles. Currently, we use the `user_defined_measures`
to store these results (budget, predictions, probabilities over the cycles). As a result, we needed to modify the 
handling of list via `_to_dict` in the file `openml.runs.run.py`. Particularly, we replace lists with the value `0`. 
Results are not contained in the `.xml` file as description of the run.
- The datasplits are again not available for active learning tasks, likely due to a switch of the test openml to 
another branch.
- Where do we can add outputs/results specific to a certain learning task?
- How are flows compared to each other or matched?




## Open Issues to be Discussed with Scikit-ActiveML
- How to deal with utility scores and subsampling? 
"""Tuner class. Currently a minimum working demo."""

import copy
import typing
import uuid
from pathlib import Path

import hyperopt

import matchzoo as mz
from matchzoo import engine


class Tuner(object):
    """
    Model parameter auto tuner.

    :param model: Model to tune. `model.params` should be complete.
    :param train_data: Training data to use. Either a preprocessed `DataPack`,
        or a `DataGenerator`.
    :param test_data: Testing data to use. A preprocessed `DataPack`.
    :param fit_kwargs: Extra keyword arguments to pass to `fit`. e.g.
        `{'batch_size': 32, 'epochs': 20}`.
    :param evaluate_kwargs: Extra keyword arguments to pass to `evaluate`. e.g.
        `{'batch_size': 32, 'epochs': 20}`.
    """

    def __init__(
        self,
        model: mz.engine.BaseModel,
        train_data: typing.Union[mz.DataPack, mz.DataGenerator],
        test_data: mz.DataPack,
        fit_kwargs: typing.Optional[dict] = None,
        evaluate_kwargs: typing.Optional[dict] = None,
        metric: typing.Union[str, mz.engine.BaseMetric] = None,
        mode: str = 'maximize',
        num_evals: int = 32,
        save_dir: typing.Union[str, Path] = mz.USER_TUNED_MODELS_DIR
    ):
        """Tuner."""
        if evaluate_kwargs is None:
            evaluate_kwargs = {}
        if fit_kwargs is None:
            fit_kwargs = {}
        self._validate_model(model)
        if metric is None:
            metric = model.params['task'].metrics[0]
        self._validate_train_data(train_data)
        self._validate_test_data(test_data)
        self._validate_kwargs(fit_kwargs)
        self._validate_kwargs(evaluate_kwargs)
        self._validate_mode(mode)
        self._validate_metric(model, metric)

        self._model = model
        self._train_data = train_data
        self._test_data = test_data
        self._fit_kwargs = fit_kwargs
        self._evaluate_kwargs = evaluate_kwargs
        self._metric = metric
        self._mode = mode
        self._num_evals = num_evals
        self._save_dir = save_dir

    def tune(self):
        """Tune."""
        orig_params = copy.deepcopy(self._model.params)

        trials = hyperopt.Trials()

        hyperopt.fmin(
            fn=self._test_func,
            space=self._model.params.hyper_space,
            algo=hyperopt.tpe.suggest,
            max_evals=self._num_evals,
            trials=trials
        )

        self._model.params = orig_params
        return self._format_trials(trials)

    def _test_func(self, space):
        self._load_space(space)
        self._model.build()
        self._model.compile()
        self._fit_model()
        results = self._eval_model()
        loss = self._get_loss(results)
        model_id = str(uuid.uuid4())
        self._save_model(model_id)
        return {
            'loss': loss,
            'space': space,
            'status': hyperopt.STATUS_OK,
            'model_id': model_id,
        }

    def _save_model(self, model_id):
        self._model.save(self._save_dir.joinpath(model_id))
        return model_id

    def _load_space(self, space):
        for key, value in space.items():
            self._model.params[key] = value

    def _get_loss(self, results):
        loss = results[self._metric]
        if self._mode == 'maximize':
            loss = -loss
        return loss

    def _eval_model(self):
        if isinstance(self._test_data, mz.DataPack):
            x, y = self._test_data.unpack()
            results = self._model.evaluate(x, y, **self._evaluate_kwargs)
        else:
            raise ValueError
        return results

    def _fit_model(self):
        if isinstance(self._train_data, mz.DataPack):
            x, y = self._train_data.unpack()
            self._model.fit(x, y, **self._fit_kwargs)
        elif isinstance(self._train_data, mz.DataGenerator):
            self._model.fit_generator(self._train_data, **self._fit_kwargs)
        else:
            raise ValueError

    def _format_trials(self, trials):
        def _format_one_trial(trial):
            metric = trial['result']['loss']
            if self._mode == 'maximize':
                metric = -metric
            return {
                'model_id': trial['result']['model_id'],
                'metric': metric,
                'sample': trial['result']['space'],
            }

        return {
            'best': _format_one_trial(trials.best_trial),
            'trials': [_format_one_trial(trial) for trial in trials.trials]
        }

    @classmethod
    def _validate_model(cls, model):
        if not isinstance(model, engine.BaseModel):
            raise TypeError
        elif not model.params.hyper_space:
            raise ValueError("Model hyper space empty.")

    @classmethod
    def _validate_train_data(cls, train_data):
        if not isinstance(train_data, (mz.DataPack, mz.DataGenerator)):
            raise TypeError

    @classmethod
    def _validate_test_data(cls, test_data):
        if not isinstance(test_data, mz.DataPack):
            raise TypeError

    @classmethod
    def _validate_metric(cls, model, metric):
        if metric not in model.params['task'].metrics:
            raise ValueError('Model does not have the metric.')

    @classmethod
    def _validate_mode(cls, mode):
        if mode not in ('maximize', 'minimize'):
            raise ValueError

    @classmethod
    def _validate_kwargs(cls, kwargs):
        if not isinstance(kwargs, dict):
            raise TypeError

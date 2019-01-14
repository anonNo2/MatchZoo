"""Tuner class. Currently a minimum working demo."""

import copy
import typing
import logging

import hyperopt

import matchzoo as mz
from matchzoo import engine
from matchzoo.auto.tuner.callbacks import Callback


class Tuner(object):
    """
    Model parameter auto tuner.

    `model.params.hyper_space` reprensents the model's hyper-parameters
    search space, which is the cross-product of individual hyper parameter's
    hyper space. When a `Tuner` builds a model, for each hyper parameter in
    `model.params`, if the hyper-parameter has a hyper-space, then a sample
    will be taken in the space. However, if the hyper-parameter does not
    have a hyper-space, then the default value of the hyper-parameter will
    be used.

    :param params: A completed parameter table to tune. Usually `model.params`
        of the desired model to tune. `params.completed()` should be `True`.
    :param train_data: Training data to use. Either a preprocessed `DataPack`,
        or a `DataGenerator`.
    :param test_data: Testing data to use. A preprocessed `DataPack`.
    :param fit_kwargs: Extra keyword arguments to pass to `fit`.
        (default: `dict(epochs=10, batch_size=32, verbose=0)`)
    :param evaluate_kwargs: Extra keyword arguments to pass to `evaluate`.
        (default: `dict(batch_size=1024, verbose=0)`)
    :param metric: Metric to tune upon. Must be one of the metrics in
        `model.params['task'].metrics`. (default: the first metric in
        `params.['task'].metrics`.
    :param mode: Either `maximize` the metric or `minimize` the metric.
        (default: 'maximize')
    :param num_runs: Number of runs. Each run takes a sample in
        `params.hyper_space` and build a model based on the sample.
        (default: 10)
    :param callbacks: A list of callbacks to handle.

    Example:
        >>> import matchzoo as mz
        >>> train = mz.datasets.toy.load_data('train')
        >>> dev = mz.datasets.toy.load_data('dev')
        >>> prpr = mz.models.DenseBaseline.get_default_preprocessor()
        >>> train = prpr.fit_transform(train, verbose=0)
        >>> dev = prpr.transform(dev, verbose=0)
        >>> model = mz.models.DenseBaseline()
        >>> model.params['input_shapes'] = prpr.context['input_shapes']
        >>> model.params['task'] = mz.tasks.Ranking()
        >>> tuner = mz.auto.Tuner(
        ...     params=model.params,
        ...     train_data=train,
        ...     test_data=dev,
        ...     num_runs=2
        ... )
        >>> results = tuner.tune()
        >>> sorted(results['best'].keys())
        ['#', 'params', 'sample', 'score']

    """

    def __init__(
        self,
        params: mz.engine.ParamTable,
        train_data: typing.Union[mz.DataPack, mz.DataGenerator],
        test_data: mz.DataPack,
        fit_kwargs: dict = None,
        evaluate_kwargs: dict = None,
        metric: typing.Union[str, mz.engine.BaseMetric] = None,
        mode: str = 'maximize',
        num_runs: int = 10,
        callbacks: typing.List[Callback] = None
    ):
        """Tuner."""
        fit_kwargs = fit_kwargs or dict(epochs=10, batch_size=32, verbose=0)
        evaluate_kwargs = evaluate_kwargs or dict(batch_size=1024, verbose=0)
        callbacks = callbacks or []

        self._validate_params(params)
        metric = metric or params['task'].metrics[0]
        self._validate_train_data(train_data)
        self._validate_test_data(test_data)
        self._validate_kwargs(fit_kwargs)
        self._validate_kwargs(evaluate_kwargs)
        self._validate_mode(mode)
        self._validate_metric(params, metric)
        self._validate_callbacks(callbacks)

        self._params = copy.deepcopy(params)
        self._train_data = train_data
        self._test_data = test_data
        self._fit_kwargs = fit_kwargs
        self._evaluate_kwargs = evaluate_kwargs
        self._metric = metric
        self._mode = mode
        self._num_runs = num_runs
        self._callbacks = callbacks

        self.__curr_run_num = 0

    def tune(self):
        """
        Start tuning.

        Notice that `tune` does not affect the tuner's inner state, so each
        new call to `tune` starts fresh. In other words, hyperspaces are
        suggestive only within the same `tune` call.
        """
        self.__curr_run_num = 0
        logging.getLogger('hyperopt').setLevel(logging.CRITICAL)

        trials = hyperopt.Trials()
        hyperopt.fmin(
            fn=self._run,
            space=self._params.hyper_space,
            algo=hyperopt.tpe.suggest,
            max_evals=self._num_runs,
            trials=trials
        )

        return {
            'best': trials.best_trial['result']['mz_result'],
            'trials': [trial['result']['mz_result'] for trial in trials.trials]
        }

    def _run(self, space):
        self.__curr_run_num += 1
        self._load_space(space)

        # build
        model = self._params['model_class'](params=self._params)
        model.build()
        model.compile()
        self._handle_callbacks_build_end(model)

        # fit & evaluate
        self._fit_model(model)
        lookup = self._evaluate_model(model)
        score = lookup[self._metric]

        # construct result
        result = {
            '#': self.__curr_run_num,
            'params': copy.deepcopy(self._params),
            'sample': space,
            'score': score
        }

        self._handle_callbacks_run_end(model, result)

        return {
            'loss': self._fix_loss_sign(score),
            'status': hyperopt.STATUS_OK,
            'mz_result': result
        }

    def _load_space(self, space):
        for key, value in space.items():
            self._params[key] = value

    def _handle_callbacks_build_end(self, model):
        for callback in self._callbacks:
            callback.on_build_end(model)

    def _handle_callbacks_run_end(self, model, result):
        for callback in self._callbacks:
            callback.on_run_end(model, result)

    def _fit_model(self, model):
        if isinstance(self._train_data, mz.DataPack):
            x, y = self._train_data.unpack()
            model.fit(x, y, **self._fit_kwargs)
        elif isinstance(self._train_data, mz.DataGenerator):
            model.fit_generator(self._train_data, **self._fit_kwargs)
        else:
            raise ValueError

    def _evaluate_model(self, model):
        if isinstance(self._test_data, mz.DataPack):
            x, y = self._test_data.unpack()
            results = model.evaluate(x, y, **self._evaluate_kwargs)
        else:
            raise ValueError
        return results

    def _fix_loss_sign(self, loss):
        if self._mode == 'maximize':
            loss = -loss
        return loss

    @classmethod
    def _validate_params(cls, params):
        if not isinstance(params, engine.ParamTable):
            raise TypeError
        if not params.hyper_space:
            raise ValueError("Parameter hyper-space empty.")
        if not params.completed():
            raise ValueError("Parameters not complete.")

    @classmethod
    def _validate_train_data(cls, train_data):
        if not isinstance(train_data, (mz.DataPack, mz.DataGenerator)):
            raise TypeError

    @classmethod
    def _validate_test_data(cls, test_data):
        if not isinstance(test_data, mz.DataPack):
            raise TypeError

    @classmethod
    def _validate_kwargs(cls, kwargs):
        if not isinstance(kwargs, dict):
            raise TypeError

    @classmethod
    def _validate_mode(cls, mode):
        if mode not in ('maximize', 'minimize'):
            raise ValueError('`mode` should be one of `maximize`, `minimize`.')

    @classmethod
    def _validate_metric(cls, params, metric):
        if metric not in params['task'].metrics:
            raise ValueError('Target metric does not exist in the task.')

    @classmethod
    def _validate_num_runs(cls, num_runs):
        if not isinstance(num_runs, int):
            raise TypeError

    @classmethod
    def _validate_callbacks(cls, callbacks):
        for callback in callbacks:
            if not isinstance(callback, Callback):
                raise TypeError

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        self._validate_params(value)
        self._validate_metric(value, self._metric)
        self._params = value

    @property
    def train_data(self):
        return self._train_data

    @train_data.setter
    def train_data(self, value):
        self._validate_train_data(value)
        self._train_data = value

    @property
    def test_data(self):
        return self._test_data

    @test_data.setter
    def test_data(self, value):
        self._validate_test_data(value)
        self._test_data = value

    @property
    def fit_kwargs(self):
        return self._fit_kwargs

    @fit_kwargs.setter
    def fit_kwargs(self, value):
        self._validate_kwargs(value)
        self._fit_kwargs = value

    @property
    def evaluate_kwargs(self):
        return self._evaluate_kwargs

    @evaluate_kwargs.setter
    def evaluate_kwargs(self, value):
        self._validate_kwargs(value)
        self._evaluate_kwargs = value

    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, value):
        self._validate_metric(self._params, value)
        self._metric = value

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._validate_mode(value)
        self._mode = value

    @property
    def num_runs(self):
        return self._num_runs

    @num_runs.setter
    def num_runs(self, value):
        self._validate_num_runs(value)
        self._num_runs = value

    @property
    def callbacks(self):
        return self._callbacks

    @callbacks.setter
    def callbacks(self, value):
        self._validate_callbacks(value)
        self._callbacks = value

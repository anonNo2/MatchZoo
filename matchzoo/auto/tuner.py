"""Tuner class. Currently a minimum working demo."""

import copy
import uuid
from pathlib import Path

import hyperopt

import matchzoo
from matchzoo import engine


class Tuner(object):
    """Tuner."""

    def __init__(self, **kwargs):
        """Tuner."""

        self._params = engine.ParamTable()

        self._params.add(engine.Param(
            'model', validator=self._validate_model
        ))
        self._params.add(engine.Param(
            'train_data',
            validator=lambda data: isinstance(
                data, (matchzoo.DataPack, matchzoo.DataGenerator))
        ))
        self._params.add(engine.Param(
            'test_data',
            validator=lambda data: isinstance(data, matchzoo.DataPack)
        ))
        self._params.add(engine.Param(
            'fit_kwargs', {}, validator=lambda x: isinstance(x, dict)
        ))
        self._params.add(engine.Param(
            'evaluate_kwargs', {}, validator=lambda x: isinstance(x, dict)
        ))
        self._params.add(engine.Param(
            'after_build',
            validator=lambda x: callable(x)
        ))
        self._params.add(engine.Param(
            'mode', 'maximize',
            validator=lambda mode: mode in ('minimize', 'maximize')
        ))
        self._params.add(engine.Param(
            'optimizing_metric', validator=self._valiadate_metric
        ))
        self._params.add(engine.Param(
            'num_evals', 8,
            validator=lambda max_evals: isinstance(max_evals, int)
        ))
        self._params.add(engine.Param(
            'save_dir', matchzoo.USER_TUNED_MODELS_DIR,
            validator=lambda save_dir: bool(Path(save_dir))
        ))
        for key, value in kwargs:
            self._params[key] = value

    def _valiadate_metric(self, metric):
        return metric in self._params['model'].params['task'].metrics

    @classmethod
    def _validate_model(cls, model):
        if not isinstance(model, engine.BaseModel):
            return False
        elif not model.params.hyper_space:
            print("Model hyper space empty.")
            return False
        else:
            return True

    @property
    def params(self):
        """:return: tuner configuration paratmeters."""
        return self._params

    def tune(self):
        """Tune."""
        orig_params = copy.deepcopy(self._params['model'].params)

        trials = hyperopt.Trials()

        hyperopt.fmin(
            fn=self._test_func,
            space=self._params['model'].params.hyper_space,
            algo=hyperopt.tpe.suggest,
            max_evals=self._params['num_evals'],
            trials=trials
        )

        self._params['model'].params = orig_params
        mode = self._params['mode']
        return self._clean_up_trials(trials, mode)

    def _test_func(self, space):
        model = self._prepare_model(space)

        score = self._eval(
            model=model,
            train_data=self._params['train_data'],
            test_data=self._params['test_data'],
            metric=self._params['optimizing_metric'],
            mode=self._params['mode'],
            fit_kwargs=self._params['fit_kwargs'],
            evaluate_kwargs=self._params['evaluate_kwargs'],
        )

        model_id = str(uuid.uuid4())
        self._params['model'].save(self._params['save_dir'].joinpath(model_id))

        return {
            'loss': score,
            'space': space,
            'status': hyperopt.STATUS_OK,
            'model_id': model_id,
            'model_params': self._params['model'].params
        }

    def _prepare_model(self, space):
        for key, value in space.items():
            self._params['model'].params[key] = value
        model = self._params['model']
        model.build()
        model.compile()
        if self._params['after_build']:
            self._params['after_build'](model)
        return model

    @classmethod
    def _eval(cls, model, train_data, test_data, metric, mode,
              fit_kwargs, evaluate_kwargs):
        if isinstance(train_data, matchzoo.DataPack):
            model.fit(*train_data.unpack(), **fit_kwargs)
        elif isinstance(train_data, matchzoo.DataGenerator):
            model.fit_generator(train_data, **fit_kwargs)
        else:
            raise ValueError

        if isinstance(test_data, matchzoo.DataPack):
            results = model.evaluate(*test_data.unpack(), **evaluate_kwargs)
        elif isinstance(test_data, matchzoo.DataGenerator):
            results = model.evaluate_(test_data, **evaluate_kwargs)
        else:
            raise ValueError

        score = results[metric]
        if mode == 'maximize':
            score = -score
        return score

    @classmethod
    def _clean_up_trials(cls, trials, mode):
        def _format_trial(trial):
            score = trial['result']['loss']
            if mode == 'maximize':
                score = -score
            return {
                'model_id': trial['result']['model_id'],
                'model_params': trial['result']['model_params'],
                'metric': score,
                'sampled_params': trial['result']['space'],
            }

        return {
            'best': _format_trial(trials.best_trial),
            'trials': [_format_trial(trial) for trial in trials.trials]
        }

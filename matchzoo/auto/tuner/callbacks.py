import uuid

import matchzoo as mz


class Callback(object):
    def on_build_end(self, model: mz.engine.BaseModel):
        pass

    def on_run_end(self, model: mz.engine.BaseModel, result: dict):
        pass


class LambdaCallback(Callback):
    def __init__(self, on_build_end, on_result_end):
        self._on_build_end = on_build_end
        self._on_result_end = on_result_end

    def on_build_end(self, model: mz.engine.BaseModel):
        self._on_build_end(model)

    def on_run_end(self, model: mz.engine.BaseModel, result: dict):
        self._on_result_end(model, result)


class SaveModel(Callback):
    """Save trained model to `matchzoo.USER_TUNED_MODELS_DIR`."""

    def on_run_end(self, model: mz.engine.BaseModel, result: dict):
        model_id = str(uuid.uuid4())
        model.save(mz.USER_TUNED_MODELS_DIR.joinpath(model_id))
        result['model_id'] = model_id


class LoadEmbeddingMatrix(Callback):
    """Load a pre-trained embedding after the model is built."""

    def __init__(self, embedding_matrix):
        self._embedding_matrix = embedding_matrix

    def on_build_end(self, model: mz.engine.BaseModel):
        model.load_embedding_matrix(self._embedding_matrix)


class LogResult(Callback):
    def on_run_end(self, model: mz.engine.BaseModel, result: dict):
        print(f"Run #{result['#']}")
        print(f"Score: {result['score']}")
        print(f"Sampled hyper-space: {result['sample']}")
        print()

from typing import NamedTuple
from kfp.v2.dsl import (
    Artifact, 
    Dataset, 
    Input,
    InputPath, 
    Model, 
    Output,
    OutputPath,
    Metrics,
    HTML,
    component
)

def get_champion_model(
    project_id: str,
    project_name: str,
    runner:str,
    location: str,
    champion_unique_catalog_ids_dataset: Output[Dataset],
    champion_unique_user_ids_dataset: Output[Dataset],
    # champion_cg_model_object: Output[Model],
) -> NamedTuple(
    "Outputs",
    [
        ("is_cold_start", bool),
        ("cg_model_weight_path", str),
        ("embedding_dimension", int),
    ],
):
    import pandas_gbq
    import json

    model_league_path = f'MLOPS_TRACKING.{runner}_model_league_{project_name}'
    production_query = f"""
    SELECT * FROM `{model_league_path}`
    """
    model_league = pandas_gbq.read_gbq(
        query=production_query, 
        project_id=project_id,
        use_bqstorage_api=True,
        location=location,
    )

    if len(model_league) > 0:
        model_league = model_league[model_league['tag']=='production'].reset_index()
        model_path = model_league['cg_model_path'][0].replace('gs://', '')
        cg_model_weight_path = '/'.join(model_path.split('/')[1:-1])

        with open(champion_unique_catalog_ids_dataset.path, 'w') as outfile: outfile.write(json.dumps(model_league['unique_catalog_ids'][0]));
        with open(champion_unique_user_ids_dataset.path, 'w') as outfile: outfile.write(json.dumps(model_league['unique_user_ids'][0]));
        embedding_dimension = int(model_league['embedding_dimension'][0])
        is_cold_start = False
    else:
        cg_model_weight_path = ''
        with open(champion_unique_catalog_ids_dataset.path, 'w') as outfile: outfile.write(json.dumps([]));
        with open(champion_unique_user_ids_dataset.path, 'w') as outfile: outfile.write(json.dumps([]));
        embedding_dimension = 32
        is_cold_start = True
    
    return (
        is_cold_start,
        cg_model_weight_path,
        embedding_dimension,
    )


def candidate_generation_train(
    is_cold_start: bool,
    bucket_name: str,
    model_params: str,
    user_id_col: str,
    catalog_col: str,
    embedding_dimension: int,
    champion_embedding_dimension: int,
    cg_model_weight_path: str,
    catalog_dataset: Input[Dataset],
    train_dataset: Input[Dataset],
    test_dataset: Input[Dataset],
    champion_unique_catalog_ids_dataset: Input[Dataset],
    champion_unique_user_ids_dataset: Input[Dataset],
    chosen_unique_catalog_ids_dataset: Output[Dataset],
    chosen_unique_user_ids_dataset: Output[Dataset],
    cg_metrics_object: Output[Metrics],
    cg_model_object: Output[Model],
    cg_index_object: Output[Artifact],
) -> NamedTuple(
    "Outputs",
    [
        ("best_model", str),
        ("chosen_embedding_dimension", int),
    ],
):
    import tensorflow as tf
    import tensorflow_recommenders as tfrs
    from google.cloud import storage
    import pandas as pd
    import json
    import ast
    import os

    model_params = json.loads(model_params)

    catalog = pd.read_parquet(catalog_dataset.path)
    catalog_tf_dataset = tf.data.Dataset.from_tensor_slices(dict(catalog))
    catalog_tf_dataset = catalog_tf_dataset.map(lambda x: x[catalog_col])

    train = pd.read_parquet(train_dataset.path)
    train_tf_dataset = tf.data.Dataset.from_tensor_slices(dict(train))
    train_tf_dataset = train_tf_dataset.map(
        lambda x: {col: x[col] for col in train.columns}
    )

    test = pd.read_parquet(test_dataset.path)
    test_tf_dataset = tf.data.Dataset.from_tensor_slices(dict(test))
    test_tf_dataset = test_tf_dataset.map(
        lambda x: {col: x[col] for col in test.columns}
    )

    if not os.path.exists(cg_model_weight_path):
        os.makedirs(cg_model_weight_path)

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name.replace('gs://', ''))

    blobs = storage_client.list_blobs(
        bucket_name.replace('gs://', ''),
        prefix=cg_model_weight_path,
    )
    for blob in blobs:
        if 'cg_model_object' in blob.name or 'checkpoint' in blob.name:
            blob = bucket.blob(blob.name)
            blob.download_to_filename(blob.name)

    unique_catalog_ids = catalog[catalog_col].tolist()
    unique_user_ids = train[user_id_col].unique().tolist()
    with open(champion_unique_catalog_ids_dataset.path, 'r') as openfile: champion_unique_catalog_ids = ast.literal_eval(json.load(openfile));
    with open(champion_unique_user_ids_dataset.path, 'r') as openfile: champion_unique_user_ids = ast.literal_eval(json.load(openfile));

    def TowerGeneration(
        unique_catalog_ids,
        unique_user_ids,
        embedding_dimension,
    ):
        # Define query and candidate towers.
        query_tower = tf.keras.Sequential([
          tf.keras.layers.StringLookup(
              vocabulary=unique_user_ids, mask_token=None),
          # We add an additional embedding to account for unknown tokens.
          tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
        ])

        candidate_tower = tf.keras.Sequential([
          tf.keras.layers.StringLookup(
              vocabulary=unique_catalog_ids, mask_token=None),
          tf.keras.layers.Embedding(len(unique_catalog_ids) + 1, embedding_dimension)
        ])

        # Define your objectives.
        task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
            catalog_tf_dataset.batch(128).map(candidate_tower)
          )
        )
        return {
            'query_tower': query_tower,
            'candidate_tower': candidate_tower,
            'task': task,
        }

    class CandidateGeneration(tfrs.Model):
        # We derive from a custom base class to help reduce boilerplate. Under the hood,
        # these are still plain Keras Models.
        def __init__(
            self,
            query_tower: tf.keras.Model,
            candidate_tower: tf.keras.Model,
            task: tfrs.tasks.Retrieval
        ):
            super().__init__()

            # Set up user and catalog representations.
            self.query_tower = query_tower
            self.candidate_tower = candidate_tower

            # Set up a retrieval task.
            self.task = task

        def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
            # Define how the loss is computed.

            user_embeddings = self.query_tower(features[user_id_col])
            catalog_embeddings = self.candidate_tower(features[catalog_col])

            return self.task(user_embeddings, catalog_embeddings)

    # Challenger CG
    cg_model = CandidateGeneration(
        **TowerGeneration(
            unique_catalog_ids,
            unique_user_ids,
            embedding_dimension,
        )
    )
    cg_model.compile(optimizer=tf.keras.optimizers.Adagrad(model_params['learning_rate']))

    cg_history = cg_model.fit(
        train_tf_dataset.batch(8192).cache(),
        epochs=model_params['epochs']
    )
    cg_model.save_weights(cg_model_object.path)
    model_eval_metrics = cg_model.evaluate(test_tf_dataset.batch(4096).cache(), return_dict=True)
    for metric, score in model_eval_metrics.items(): cg_metrics_object.log_metric(f'challenger_{metric}', score);

    # Champion CG
    if is_cold_start:
        model_champion_eval_metrics = {key: 0 for (key, val) in model_eval_metrics.items()}
    else:
        champion_cg_model = CandidateGeneration(
            **TowerGeneration(
                champion_unique_catalog_ids,
                champion_unique_user_ids,
                champion_embedding_dimension,
            )
        )
        champion_cg_model.compile(optimizer=tf.keras.optimizers.Adagrad(model_params['learning_rate']))
        champion_cg_model.load_weights(cg_model_weight_path + '/cg_model_object')
        model_champion_eval_metrics = champion_cg_model.evaluate(test_tf_dataset.batch(4096).cache(), return_dict=True)
    for metric, score in model_champion_eval_metrics.items(): cg_metrics_object.log_metric(f'champion_{metric}', score);

    # Compare and select best model
    eval_comparison = []
    selected_eval_metrics = {key: val for (key, val) in model_eval_metrics.items() if key.startswith('factorized')}
    for metric, score in selected_eval_metrics.items():
        if score >= model_champion_eval_metrics[metric]:
            is_new_model_win = True
        else:
            is_new_model_win = False
        eval_comparison.append(is_new_model_win)

    # if new model wins
    if all(eval_comparison):
        best_model = 'challenger'
        best_model_obj = cg_model
        with open(chosen_unique_catalog_ids_dataset.path, 'w') as outfile: outfile.write(json.dumps(unique_catalog_ids));
        with open(chosen_unique_user_ids_dataset.path, 'w') as outfile: outfile.write(json.dumps(unique_user_ids));
        chosen_embedding_dimension = embedding_dimension

    # if new model loses
    else:
        best_model = 'champion'
        best_model_obj = champion_cg_model
        with open(chosen_unique_catalog_ids_dataset.path, 'w') as outfile: outfile.write(json.dumps(champion_unique_catalog_ids));
        with open(chosen_unique_user_ids_dataset.path, 'w') as outfile: outfile.write(json.dumps(champion_unique_user_ids));
        chosen_embedding_dimension = champion_embedding_dimension

    #initialize the scann layer
    scann_index = tfrs.layers.factorized_top_k.ScaNN(best_model_obj.query_tower)
    scann_index.index_from_dataset(
      tf.data.Dataset.zip(
          (
              catalog_tf_dataset.batch(100), 
              catalog_tf_dataset.batch(100).map(best_model_obj.candidate_tower)
          )
      )
    )
    #build the scann layer
    scann_index(tf.constant(["0"]))

    # Save the index.
    tf.saved_model.save(
      scann_index,
      cg_index_object.path,
      options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"])
    )

    return (
        best_model,
        chosen_embedding_dimension,
    )

def candidate_generation_prediction(
    user_id_col: str,
    catalog_col: str,
    model_params: str,
    user_dataset: Input[Dataset],
    cg_index_object: Input[Artifact],
    cg_results_dataset: Output[Dataset],
):
    from joblib import Parallel, delayed
    import multiprocessing as mp
    from tqdm import tqdm
    import pandas as pd
    import numpy as np
    import json

    import tensorflow as tf
    import tensorflow_recommenders as tfrs

    class ParallelCGError(Exception):
        """ An exception class for ParallelCG """
        pass

    class ParallelCG():
        def __init__(
            self,
            df: pd.DataFrame,
            n_job: int,
            user_id_col: str,
            catalog_col: str,
            cg_top_n: str,
            index_path: str,
            # index: tfrs.layers.factorized_top_k.ScaNN,
        ):
            self.df = df
            self.n_job = n_job
            self.user_id_col = user_id_col
            self.catalog_col = catalog_col
            self.cg_top_n = cg_top_n

            self.index_path = index_path

            if not self.n_job:
                raise ParallelCGError(f'n_job is not provided. Cores range: 1-{mp.cpu_count()}')
            elif self.n_job > mp.cpu_count():
                raise ParallelCGError(f'Exceed CPU cores count. Max cores: {mp.cpu_count()}')

        def infer(
          self
        ):
            self.df_partition = np.array_split(self.df, self.n_job)
            parallelize_results = Parallel(n_jobs=self.n_job)(delayed(self.forecast)(core) for core in tqdm(range(self.n_job)))
            self.df_infer = pd.concat(parallelize_results, ignore_index=True)

        def forecast(
            self,
            core
        ):
            from tensorflow_recommenders.layers.factorized_top_k import ScaNN
            index = tf.saved_model.load(self.index_path)
            self.df_partition[core]['cg_output'] =  self.df_partition[core][self.user_id_col].apply(
                lambda row: index(tf.constant([row]))[1].numpy().flatten()[:self.cg_top_n]
            )
            return self.df_partition[core]

        def compile_result(
            self,
        ):
            product_df = pd.DataFrame(
                self.df_infer['cg_output'].tolist(), 
                columns=[f'product_{i+1}' for i in range(self.cg_top_n)]
            )
            # concat df and split_df
            self.results = pd.concat([self.df, product_df], axis=1)

            self.results = pd.melt(
                self.results, 
                id_vars=[self.user_id_col], 
                value_vars=[f'product_{i+1}' for i in range(self.cg_top_n)],
                value_name=self.catalog_col,
            ).sort_values(
                by=[
                    self.user_id_col,
                    'variable'
                ]
            )

    model_params = json.loads(model_params)
    user = pd.read_parquet(user_dataset.path)

    cg = ParallelCG(
        df=user,
        n_job=mp.cpu_count(),
        user_id_col=user_id_col,
        catalog_col=catalog_col,
        cg_top_n=model_params['top_n'],
        index_path=cg_index_object.path,
    )
    cg.infer()
    cg.compile_result()
    cg.results.to_parquet(cg_results_dataset.path, index=False)

def ranking(
    is_endpoint: bool,
    best_model: str,
    model_params: str,
    user_id_col: str,
    catalog_col: str,
    product_score: str,
    chosen_embedding_dimension: int,
    chosen_unique_catalog_ids_dataset: Input[Dataset],
    chosen_unique_user_ids_dataset: Input[Dataset],
    train_dataset: Input[Dataset],
    test_dataset: Input[Dataset],
    cg_results_dataset: Input[Dataset],
    r_metrics_object: Output[Metrics],
    r_model_object: Output[Model],
):
    from joblib import Parallel, delayed
    import multiprocessing as mp
    from tqdm import tqdm
    import numpy as np

    import tensorflow as tf
    import tensorflow_recommenders as tfrs
    import pandas as pd
    import json

    model_params = json.loads(model_params)

    train = pd.read_parquet(train_dataset.path)
    train_tf_dataset = tf.data.Dataset.from_tensor_slices(dict(train))
    train_tf_dataset = train_tf_dataset.map(
        lambda x: {col: x[col] for col in train.columns}
    )

    test = pd.read_parquet(test_dataset.path)
    test_tf_dataset = tf.data.Dataset.from_tensor_slices(dict(test))
    test_tf_dataset = test_tf_dataset.map(
        lambda x: {col: x[col] for col in test.columns}
    )

    with open(chosen_unique_catalog_ids_dataset.path, 'r') as openfile: chosen_unique_catalog_ids = json.load(openfile);
    with open(chosen_unique_user_ids_dataset.path, 'r') as openfile: chosen_unique_user_ids = json.load(openfile);

    class RankingModel(tf.keras.Model):
        def __init__(
            self,
            unique_user_ids: int,
            unique_catalog_ids: int,
            embedding_dimension: int,
        ):
            super().__init__()

            # Compute embeddings for users.
            self.user_embeddings = tf.keras.Sequential([
                tf.keras.layers.StringLookup(
                    vocabulary=unique_user_ids, mask_token=None),
                tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
            ])

            # Compute embeddings for catalogs.
            self.catalog_embeddings = tf.keras.Sequential([
                tf.keras.layers.StringLookup(
                    vocabulary=unique_catalog_ids, mask_token=None),
                tf.keras.layers.Embedding(len(unique_catalog_ids) + 1, embedding_dimension)
            ])

            # Compute predictions.
            self.ratings = tf.keras.Sequential([
                # Learn multiple dense layers.
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                # Make rating predictions in the final layer.
                tf.keras.layers.Dense(1)
            ])

        def call(self, inputs):

            user_id, catalog_id = inputs

            user_embedding = self.user_embeddings(user_id)
            catalog_embedding = self.catalog_embeddings(catalog_id)

            return self.ratings(tf.concat([user_embedding, catalog_embedding], axis=1))

    class Ranking(tfrs.models.Model):

        def __init__(
            self,
            unique_user_ids: int,
            unique_catalog_ids: int,
            embedding_dimension: int=32,
        ):
            super().__init__()
            self.ranking_model: tf.keras.Model = RankingModel(
                unique_user_ids,
                unique_catalog_ids,
                embedding_dimension,
            )
            self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
                loss = tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.RootMeanSquaredError()]
            )

        def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
            return self.ranking_model(
                (features[user_id_col], features[catalog_col]))

        def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
            labels = features.pop(product_score)

            rating_predictions = self(features)

            # The task computes the loss and the metrics.
            return self.task(labels=labels, predictions=rating_predictions)

    r_model = Ranking(
        unique_user_ids=chosen_unique_user_ids,
        unique_catalog_ids=chosen_unique_catalog_ids,
        embedding_dimension=chosen_embedding_dimension,
    )

    r_model.compile(
        optimizer=tf.keras.optimizers.Adagrad(model_params['learning_rate']),
    )

    r_history = r_model.fit(
        train_tf_dataset.batch(8192).cache(),
        epochs=model_params['epochs']
    )
    r_model.save_weights(r_model_object.path)

    model_eval_metrics = r_model.evaluate(test_tf_dataset.batch(4096).cache(), return_dict=True)
    for metric, score in model_eval_metrics.items(): r_metrics_object.log_metric(metric, score);

    if is_endpoint:
        return
    else:
        ###########################################################
        ######################## ParallelR ########################
        ###########################################################

        class ParallelRError(Exception):
            """ An exception class for ParallelR """
            pass

        class ParallelR():
            def __init__(
                self,
                path: str,
                unique_user_ids: list,
                unique_catalog_ids: list,
                embedding_dimension:int,
                df: pd.DataFrame,
                n_job: int,
                user_id_col: str,
                catalog_col: str,
            ):
                self.df = df
                self.n_job = n_job
                self.user_id_col = user_id_col
                self.catalog_col = catalog_col

                self.path = path

                self.unique_user_ids = unique_user_ids
                self.unique_catalog_ids = unique_catalog_ids
                self.embedding_dimension = embedding_dimension

                if not self.n_job:
                    raise ParallelRError(f'n_job is not provided. Cores range: 1-{mp.cpu_count()}')
                elif self.n_job > mp.cpu_count():
                    raise ParallelRError(f'Exceed CPU cores count. Max cores: {mp.cpu_count()}')

            def infer(
              self
            ):
                self.df_partition = np.array_split(self.df, self.n_job)
                parallelize_results = Parallel(n_jobs=self.n_job)(delayed(self.forecast)(core) for core in tqdm(range(self.n_job)))
                self.df_infer = pd.concat(parallelize_results, ignore_index=True)

            def forecast(
                self,
                core
            ):
                r_model = Ranking(
                    unique_user_ids=self.unique_user_ids,
                    unique_catalog_ids=self.unique_catalog_ids,
                    embedding_dimension=self.embedding_dimension,
                )
                r_model.load_weights(self.path)

                self.df_partition[core] =  self.df_partition[core].apply(
                    lambda row: self.rerank(
                        r_model,
                        row
                    ),
                    axis=1,
                )
                return self.df_partition[core]

            def rerank(
                self,
                r_model,
                row,
            ):
                score = r_model(
                    {
                        self.user_id_col: np.array([row[self.user_id_col]]),
                        self.catalog_col: np.array([row[self.catalog_col]])
                    }
                )
                row['r_score'] = score.numpy().flatten()[0]
                return row

            def compile_result(
                self,
            ):
                self.results = self.df_infer.sort_values(
                    by=[
                        self.user_id_col,
                        'r_score'
                    ],
                    ascending=[
                        True,
                        False,
                    ]
                )

        cg_results = pd.read_parquet(cg_results_dataset.path)

        ranking_params['path'] = r_model_object.path
        ranking_params['df'] = cg_results
        ranking_params['n_job'] = mp.cpu_count()
        ranking_params['user_id_col'] = user_id_col
        ranking_params['catalog_col'] = catalog_col

        r = ParallelR(**ranking_params)

        r.infer()
        r.compile_result()
        r.results.to_parquet(r_results_dataset.path, index=False)

        return
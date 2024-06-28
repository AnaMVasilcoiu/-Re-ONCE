from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import tensorflow as tf
from ebrec.evaluation import AucScore, MetricEvaluator, MrrScore, NdcgScore
from ebrec.models.newsrec.model_config import hparams_naml, hparams_nrms
from ebrec.models.newsrec.naml import NAMLModel
from ebrec.utils._articles import (
    convert_text2encoding_with_transformers,
    create_article_id_to_value_mapping,
)
from ebrec.utils._articles_behaviors import map_list_article_id_to_value
from ebrec.utils._behaviors import (
    add_known_user_column,
    add_prediction_scores,
    create_binary_labels_column,
    create_user_id_to_int_mapping,
    sampling_strategy_wu2019,
    truncate_history,
)
from ebrec.utils._constants import (
    DEFAULT_ARTICLE_ID_COL,
    DEFAULT_ARTICLE_MODIFIED_TIMESTAMP_COL,
    DEFAULT_ARTICLE_PUBLISHED_TIMESTAMP_COL,
    DEFAULT_ARTICLE_TYPE_COL,
    DEFAULT_BODY_COL,
    DEFAULT_CATEGORY_COL,
    DEFAULT_CATEGORY_STR_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_ENTITIES_COL,
    DEFAULT_GENDER_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_HISTORY_IMPRESSION_TIMESTAMP_COL,
    DEFAULT_HISTORY_READ_TIME_COL,
    DEFAULT_HISTORY_SCROLL_PERCENTAGE_COL,
    DEFAULT_IMAGE_IDS_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_IMPRESSION_TIMESTAMP_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_READ_TIME_COL,
    DEFAULT_SCROLL_PERCENTAGE_COL,
    DEFAULT_SENTIMENT_LABEL_COL,
    DEFAULT_SENTIMENT_SCORE_COL,
    DEFAULT_SESSION_ID_COL,
    DEFAULT_SUBCATEGORY_COL,
    DEFAULT_SUBTITLE_COL,
    DEFAULT_TITLE_COL,
    DEFAULT_TOPICS_COL,
    DEFAULT_TOTAL_INVIEWS_COL,
    DEFAULT_TOTAL_PAGEVIEWS_COL,
    DEFAULT_TOTAL_READ_TIME_COL,
    DEFAULT_USER_COL,
)
from ebrec.utils._nlp import get_transformers_word_embeddings
from ebrec.utils._polars import concat_str_columns, slice_join_dataframes
from ebrec.utils._python import (
    create_lookup_dict,
    create_lookup_objects,
    rank_predictions_by_score,
    time_it,
    write_submission_file,
)
from transformers import AutoModel, AutoTokenizer


@dataclass
class NewsrecDataLoader(tf.keras.utils.Sequence):
    """
    A DataLoader for news recommendation.
    """

    behaviors: pl.DataFrame
    history_column: str
    article_dict: dict[int, any]
    unknown_representation: str
    eval_mode: bool = False
    batch_size: int = 32
    inview_col: str = DEFAULT_INVIEW_ARTICLES_COL
    labels_col: str = DEFAULT_LABELS_COL
    user_col: str = DEFAULT_USER_COL
    kwargs: field(default_factory=dict) = None

    def __post_init__(self):
        """
        Post-initialization method. Loads the data and sets additional attributes.
        """
        self.lookup_article_index, self.lookup_article_matrix = create_lookup_objects(
            self.article_dict, unknown_representation=self.unknown_representation
        )
        self.unknown_index = [0]
        self.X, self.y = self.load_data()
        if self.kwargs is not None:
            self.set_kwargs(self.kwargs)

    def __len__(self) -> int:
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self):
        raise ValueError("Function '__getitem__' needs to be implemented.")

    def load_data(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        X = self.behaviors.drop(self.labels_col).with_columns(
            pl.col(self.inview_col).list.len().alias("n_samples")
        )
        y = self.behaviors[self.labels_col]
        return X, y

    def set_kwargs(self, kwargs: dict):
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass(kw_only=True)
class NAMLDataLoader(NewsrecDataLoader):
    """
    Eval mode not implemented
    """

    unknown_category_value: int = 0
    unknown_subcategory_value: int = 0
    body_mapping: dict[int, list[int]] = None
    category_mapping: dict[int, int] = None
    subcategory_mapping: dict[int, int] = None

    def __post_init__(self):
        self.title_prefix = "title_"
        self.body_prefix = "body_"
        self.category_prefix = "category_"
        self.subcategory_prefix = "subcategory_"
        (
            self.lookup_article_index_body,
            self.lookup_article_matrix_body,
        ) = create_lookup_objects(
            self.body_mapping, unknown_representation=self.unknown_representation
        )
        # if self.eval_mode:
        #     raise ValueError("'eval_mode = True' is not implemented for NAML")

        return super().__post_init__()

    def transform(self, df: pl.DataFrame) -> tuple[pl.DataFrame]:
        """
        Special case for NAML as it requires body-encoding, verticals, & subvertivals
        """
        # =>
        title = df.pipe(
            map_list_article_id_to_value,
            behaviors_column=self.history_column,
            mapping=self.lookup_article_index,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        ).pipe(
            map_list_article_id_to_value,
            behaviors_column=self.inview_col,
            mapping=self.lookup_article_index,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        )
        # =>
        body = df.pipe(
            map_list_article_id_to_value,
            behaviors_column=self.history_column,
            mapping=self.lookup_article_index_body,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        ).pipe(
            map_list_article_id_to_value,
            behaviors_column=self.inview_col,
            mapping=self.lookup_article_index_body,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        )
        # =>
        category = df.pipe(
            map_list_article_id_to_value,
            behaviors_column=self.history_column,
            mapping=self.category_mapping,
            fill_nulls=self.unknown_category_value,
            drop_nulls=False,
        ).pipe(
            map_list_article_id_to_value,
            behaviors_column=self.inview_col,
            mapping=self.category_mapping,
            fill_nulls=self.unknown_category_value,
            drop_nulls=False,
        )
        # =>
        subcategory = df.pipe(
            map_list_article_id_to_value,
            behaviors_column=self.history_column,
            mapping=self.subcategory_mapping,
            fill_nulls=self.unknown_subcategory_value,
            drop_nulls=False,
        ).pipe(
            map_list_article_id_to_value,
            behaviors_column=self.inview_col,
            mapping=self.subcategory_mapping,
            fill_nulls=self.unknown_subcategory_value,
            drop_nulls=False,
        )
        return (
            pl.DataFrame()
            .with_columns(title.select(pl.all().name.prefix(self.title_prefix)))
            .with_columns(body.select(pl.all().name.prefix(self.body_prefix)))
            .with_columns(category.select(pl.all().name.prefix(self.category_prefix)))
            .with_columns(
                subcategory.select(pl.all().name.prefix(self.subcategory_prefix))
            )
        )
    
    def __getitem__(self, idx) -> tuple[tuple[np.ndarray], np.ndarray]:
        batch_X = self.X[idx * self.batch_size : (idx + 1) * self.batch_size].pipe(
            self.transform
        )
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        # =>
        batch_y = np.array(batch_y.to_list())
        his_input_title = np.array(batch_X[self.title_prefix + self.history_column].to_list())
        his_input_body = np.array(batch_X[self.body_prefix + self.history_column].to_list())

        # =>
        pred_input_title = np.array(batch_X[self.title_prefix + self.inview_col].to_list())   
        pred_input_body = np.array(batch_X[self.body_prefix + self.inview_col].to_list())

        # =>
        his_input_title = np.squeeze(self.lookup_article_matrix[his_input_title], axis=2)
        pred_input_title = np.squeeze(self.lookup_article_matrix[pred_input_title], axis=2)
        his_input_body = np.squeeze(self.lookup_article_matrix_body[his_input_body], axis=2)
        pred_input_body = np.squeeze(self.lookup_article_matrix_body[pred_input_body], axis=2)
        # =>

        his_input_vert = np.array(batch_X[self.category_prefix + self.history_column].to_list())[:, :, np.newaxis]
        his_input_subvert = np.array(batch_X[self.subcategory_prefix + self.history_column].to_list())[:, :, np.newaxis]
        pred_input_vert = np.array(batch_X[self.category_prefix + self.inview_col].to_list())[:, :, np.newaxis]
        pred_input_subvert = np.array(batch_X[self.subcategory_prefix + self.inview_col].to_list())[:, :, np.newaxis]

        if self.eval_mode: # Added the eval_mode condition
            return (
                his_input_title,
                his_input_body,
                his_input_vert,
                his_input_subvert,
                pred_input_title,
                pred_input_body,
                pred_input_vert,
                pred_input_subvert,
            ), 
        else:
            return (
                his_input_title,
                his_input_body,
                his_input_vert,
                his_input_subvert,
                pred_input_title,
                pred_input_body,
                pred_input_vert,
                pred_input_subvert,
            ), batch_y
        

def make_data_loader(PATH_DATA, do_eval):
        
    TOKEN_COL = "tokens"
    N_SAMPLES = "n"
    BATCH_SIZE = 100
    df_articles = (
        pl.scan_parquet(PATH_DATA.joinpath("../articles.parquet"))
        .select(pl.col(DEFAULT_ARTICLE_ID_COL, DEFAULT_CATEGORY_COL, DEFAULT_BODY_COL, DEFAULT_TITLE_COL, DEFAULT_TOPICS_COL, DEFAULT_SUBTITLE_COL, DEFAULT_ARTICLE_TYPE_COL))
        .with_columns(pl.Series(TOKEN_COL, np.random.randint(0, 20, (1, 10))))
        .collect()
    )
    df_history = (
        pl.scan_parquet(PATH_DATA.joinpath("history.parquet"))
        .select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL)
        .with_columns(pl.col(DEFAULT_HISTORY_ARTICLE_ID_COL).list.tail(3))
    )
    df_behaviors = (
        pl.scan_parquet(PATH_DATA.joinpath("behaviors.parquet"))
        .select(DEFAULT_USER_COL, DEFAULT_INVIEW_ARTICLES_COL, DEFAULT_CLICKED_ARTICLES_COL)
        .with_columns(pl.col(DEFAULT_INVIEW_ARTICLES_COL).list.len().alias(N_SAMPLES))
        .join(df_history, on=DEFAULT_USER_COL, how="left")
        .collect()
        .pipe(create_binary_labels_column)
    )
    # => MAPPINGS:
    article_mapping = create_article_id_to_value_mapping(
        df=df_articles, value_col=TOKEN_COL
    )
    user_mapping = create_user_id_to_int_mapping(df=df_behaviors)
    # => NPRATIO IMPRESSION - SAME LENGTHS:
    df_behaviors_train = df_behaviors.filter(pl.col(N_SAMPLES) == pl.col(N_SAMPLES).min())
    # => FOR TEST-DATALOADER
    label_lengths = df_behaviors[DEFAULT_INVIEW_ARTICLES_COL].list.len().to_list()

    body_mapping = article_mapping
    category_mapping = create_lookup_dict(
        df_articles.select(pl.col(DEFAULT_CATEGORY_COL).unique()).with_row_index(
            "row_nr"
        ),
        key=DEFAULT_CATEGORY_COL,
        value="row_nr",
    )
    subcategory_mapping = category_mapping

    dataloader = NAMLDataLoader(
        behaviors=df_behaviors_train,
        eval_mode=do_eval,
        article_dict=article_mapping,
        body_mapping=body_mapping,
        category_mapping=category_mapping,
        unknown_representation="zeros",
        subcategory_mapping=subcategory_mapping,
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        batch_size=BATCH_SIZE,
    )

    return dataloader, df_behaviors_train


# LOAD DATA:
PATH_DATA = Path("ebnerd-benchmark/data/train")
do_eval = False
train_dataloader, df_behaviors_train = make_data_loader(PATH_DATA, do_eval)


# LOAD DATA:
PATH_DATA = Path("ebnerd-benchmark/data/validation")
do_eval = True
val_dataloader, df_behaviors_val = make_data_loader(PATH_DATA, do_eval)


DEFAULT_TITLE_SIZE = 10 #30
DEFAULT_BODY_SIZE = 10 #40
UNKNOWN_TITLE_VALUE = [0] * DEFAULT_TITLE_SIZE
UNKNOWN_BODY_VALUE = [0] * DEFAULT_BODY_SIZE

DEFAULT_DOCUMENT_SIZE = 768


class hparams_naml:
    # INPUT DIMENTIONS:
    title_size: int = DEFAULT_TITLE_SIZE
    history_size: int = 3 #50
    body_size: int = DEFAULT_BODY_SIZE
    vert_num: int = 100
    vert_emb_dim: int = 10
    subvert_num: int = 100
    subvert_emb_dim: int = 10
    # MODEL ARCHITECTURE
    dense_activation: str = "relu"
    cnn_activation: str = "relu"
    attention_hidden_dim: int = 200
    filter_num: int = 400
    window_size: int = 3
    # MODEL OPTIMIZER:
    optimizer: str = "adam"
    loss: str = "cross_entropy_loss"
    dropout: float = 0.2
    learning_rate: float = 0.0001


config = hparams_naml()

# Model
TRANSFORMER_MODEL_NAME = "FacebookAI/xlm-roberta-base"
transformer_model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
word2vec_embedding = get_transformers_word_embeddings(transformer_model)

model = NAMLModel(hparams=config, word2vec_embedding=word2vec_embedding)
# model.model.summary()
model.model.fit(train_dataloader)


pred_validation = model.model.predict(val_dataloader)


df_validation = add_prediction_scores(df_behaviors_val, pred_validation.tolist()).pipe(
    add_known_user_column, known_users=df_behaviors_val[DEFAULT_USER_COL]
)

metrics = MetricEvaluator(
    labels=df_validation["labels"].to_list(),
    predictions=df_validation["scores"].to_list(),
    metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)],
)
print("Metrics:", metrics.evaluate())

import json
import os

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from UniTok import UniDep


class EbnerdPrompter:
    def __init__(self, data_path):
        self.data_path = data_path

        self.news_df = pd.read_parquet(data_path)

        self.keys = dict(
            title="title",
            subtitle="subtitle",
            category_str="category",
            topics="topics",
            body="body",
        )

        self._news_list = None
        self._news_dict = None

    def stringify(self):
        if self._news_list is not None:
            return self._news_list
        self._news_list = []

        for _, news in self.news_df.iterrows():
            string = ""
            for key in self.keys:
                string += f"[{key}] {news[self.keys[key]]}\n"
            self._news_list.append((news["article_id"], string))
        return self._news_list

    def get_news_dict(self):
        if self._news_dict is not None:
            return self._news_dict
        self._news_dict = {}
        for _, news in tqdm(self.news_df.iterrows()):
            self._news_dict[news["article_id"]] = news["title"]
        return self._news_dict

    def get_news_dict_with_category(self):
        if self._news_dict is not None:
            return self._news_dict
        self._news_dict = {}
        for _, news in tqdm(self.news_df.iterrows()):
            self._news_dict[news["article_id"]] = f'({news["category_str"]}) {news["title"]}'
        return self._news_dict


class EbnerdUser:
    def __init__(self, history_path, behaviors_path, ebnerd_prompter):
        # self.news_dict = ebnerd_prompter.get_news_dict()

        self.history_df = pd.read_parquet(history_path)
        self.behaviors_df = pd.read_parquet(behaviors_path)
        self.news_df = ebnerd_prompter.news_df

        self.user_keys = dict(
            gender="gender",
            postcode="postcode",
            age="age",
        )

        self._user_list = None

        self.postcode_mapping = {
            0: "metropolitan",
            1: "rural district",
            2: "municipality",
            3: "provincial",
            4: "big city"
        }

    def stringify(self):
        if self._user_list is not None:
            return self._user_list
        self._user_list = []

        for _, history in tqdm(self.history_df.iloc[:, :].iterrows()):
            string = ""

            # Add user characteristics (from behavior)
            behavior = self.behaviors_df.loc[self.behaviors_df['user_id'] == history['user_id']].iloc[0]

            for key in self.user_keys:
               val = behavior[self.user_keys[key]]
               if not np.isnan(val):
                    if key == "gender":
                        key_string = "Male" if val == 0 else "Female"
                    elif key == "postcode":
                        key_string = self.postcode_mapping.get(val, "Unknown")
                    else:
                        key_string = behavior[self.user_keys[key]]

                    string += f"[{key}] {key_string}\n"

            # Add clicked articles by user (from history)
            articles_sorted = sorted(zip(history['article_id_fixed'], history['read_time_fixed'], history['scroll_percentage_fixed']), key=lambda x: x[1])
            string += "[articles]"

            for i, (article_id, read_time, scroll_percentage) in enumerate(articles_sorted[:min(100, len(articles_sorted))]):
                # Add article title (TODO: enhanced title and/or subtitle/category/topics)
                article = self.news_df.loc[self.news_df['article_id'] == article_id]
                if not article.empty:
                    article = article.iloc[0]
                    string += f"({i + 1})title: {article.title}, read time: {read_time}, scroll percentage: {scroll_percentage}\n"

            self._user_list.append((history["user_id"], string))
        return self._user_list

class EbnerdColdUser:
    def __init__(self, history_path, ebnerd_prompter):
        self.news_dict = ebnerd_prompter.get_news_dict_with_category()

        self.history_df = pd.read_parquet(history_path)
        self._user_list = None

    def stringify(self):
        if self._user_list is not None:
            return self._user_list
        self._user_list = []

        for _, history in tqdm(self.history_df.iterrows()):
            string = ""

            if len(history['article_id_fixed']) > 10:
                continue
            for i, article_id in enumerate(history['article_id_fixed']):
                string += f"({i + 1}) {self.news_dict[article_id]}\n"

            self._user_list.append((history["user_id"], string))
        return self._user_list

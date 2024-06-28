import os

import numpy as np
from UniTok import UniDep

news = UniDep(os.path.join("ebnerd-benchmark/data/ebnerd_demo/tokenized_bert", 'news'))
news_llama = UniDep('ebnerd-benchmark/data/ebnerd_demo/tokenized_llama/news-llama')

news.rename_col('title', 'title-bert')
news.rename_col('subtitle', 'subtitle-bert')
news.rename_col('body', 'body-bert')
news.rename_col('category', 'category-token')
news.rename_col('caption', 'caption-token')

news_llama.rename_col('title', 'title-llama')
news_llama.rename_col('subtitle', 'subtitle-llama')
news_llama.rename_col('body', 'body-llama')
news_llama.rename_col('category', 'category-llama')
news_llama.rename_col('caption', 'caption-llama')

news.inject(news_llama, ['title-llama', 'subtitle-llama', 'body-llama', 'category-llama', 'caption-llama'])
news.export('ebnerd-benchmark/data/ebnerd_demo/news_fusion')
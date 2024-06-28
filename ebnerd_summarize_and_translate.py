from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import argparse
import os

# Translator
tokenizer_translator = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-da-en")
model_translator = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-da-en")

# Summarizer 
tokenizer_summarizer = AutoTokenizer.from_pretrained("Danish-summarisation/DanSumT5-base")
model_summarizer = AutoModelForSeq2SeqLM.from_pretrained("Danish-summarisation/DanSumT5-base")

def translate_text(text):
    inputs = tokenizer_translator.encode(text, return_tensors="pt")
    outputs = model_translator.generate(inputs, max_length=50, num_beams=4, early_stopping=True)
    translated_text = tokenizer_translator.decode(outputs[0], skip_special_tokens=True)
    return translated_text

def translate_list(text_list):
    return [translate_text(text) for text in text_list]

def summarize_text(text):
    inputs = tokenizer_summarizer(text, return_tensors="pt", max_length=512, truncation=True)
    min_length = min(100, len(text.split()))
    summary_ids = model_summarizer.generate(inputs["input_ids"], max_length=400, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer_summarizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def process_articles(df):
    # Summarize and translate the articles
    articles_cols_translate = ["title", "subtitle", "category_str"]
    for col in articles_cols_translate:
        if col == 'body':
            df[col] = df[col].apply(summarize_text)
        df[col] = df[col].apply(translate_text)
    
    df["topics"] = df["topics"].apply(translate_list)

    return df

def main():
    # Load the articles
    articles = pd.read_parquet("ebnerd-benchmark/data/ebnerd_small/articles.parquet")

    # Process the specified range of articles
    processed_articles = process_articles(articles)

    # Ensure the results directory exists
    os.makedirs("translation_results_small", exist_ok=True)

    # Save the processed articles to a new Parquet file
    output_filename = f"ebnerd-benchmark/data/ebnerd_small/summarized_translated_articles.parquet"
    processed_articles.to_parquet(output_filename)

if __name__ == "__main__":
    main()
import time
import os

from tqdm import tqdm

from utils.openai.chat_service import ChatService

from processor.ebnerd.prompter import EbnerdPrompter

MIN_INTERVAL = 1.5

# concise

news_list = EbnerdPrompter("ebnerd-benchmark/data/ebnerd_small/articles.parquet").stringify()

system = """You are asked to act as a news title enhancer. I will provide you a piece of news, with its original title, subtitle, category, topics and body. The news format is as below:

[title] {title}
[subtitle] {subtitle}
[category] {category}
[topics] {topics}
[body] {body}

where {title}, {subtitle}, {category}, {topics} and {body} will be filled with content. You can only respond with a rephrased news title which should be clear, complete, objective and neutral. You can expand the title according to the above requirements. You are not allowed to respond with any other words or explanations. Your response format should be:

[newtitle] {newtitle}

where {newtitle} should be filled with the enhanced title. Now, your role of a news title enhancer formally begins. Any other information should not disturb your role."""

save_path = "ebnerd-benchmark/data/small_title_enhanced.log"

# Create the file if it doesn't exist
if not os.path.exists(save_path):
    with open(save_path, "w"):
        pass  

exist_set = set()
with open(save_path, "r") as f:
    for line in f:
        if line and line.startswith("N"):
            exist_set.add(line.split("\t")[0])


for article_id, content in tqdm(news_list):
    start_time = time.time()
    if article_id in exist_set:
        continue

    try:
        service = ChatService(system)
        enhanced = service.ask(content)
        enhanced = enhanced.rstrip("\n")

        with open(save_path, "a") as f:
            f.write(f"{article_id}\t{enhanced}\n")
    except Exception as e:
        print(e)

    interval = time.time() - start_time
    if interval <= MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - interval)

import json
import time
import os
from tqdm import tqdm

from processor.ebnerd.prompter import EbnerdPrompter, EbnerdColdUser

from utils.openai.chat_service import ChatService

MIN_INTERVAL = 0

# concise

ebnerd_prompter = EbnerdPrompter("data/eb-nerd/ebnerd_small/articles.parquet")
history_path = "data/eb-nerd/ebnerd_small/train/history.parquet"
behaviors_path = "data/eb-nerd/ebnerd_small/train/behaviors.parquet"

user_list = EbnerdColdUser(history_path, ebnerd_prompter).stringify()[:5]

system = """You are asked to capture user's interest based on his/her browsing history, and generate a piece of news that he/she may be interested in. The format of the history is as below:

(1) (the category of the first news) the title of the first news
...
(n) (the category of the n-th news) the title of the n-th news

You can only generate a piece of news (only one) in the following json format:

{"title": <title>, "subtitle": <news subtitle>, "topics": <news topics>, "body": <news body>}

where <news topics> is limited to the following options:

(1) crime and safety
(2) entertainment
(3) media
(4) finance
(5) sports
(6) business
(7) politics
(8) health and well-being
(9) transport
(10) science and technology
(11) education
(12) conflict and disasters
(13) lifestyle and society
(14) travelling
(15) nature and environment
(16) housing and real estate
(17) art and culture
(18) events

"title", "subtitle", "topics" and "body" should be the only keys in the json dict. The news should be diverse, that is not too similar with the original provided news list. You are not allowed to respond with any other words, explanations or notes. ONLY GIVE ME JSON-FORMAT NEWS. Now, the task formally begins. Any other information should not disturb you."""

save_path = "data/eb-nerd/news_generator.log"

# Create the file if it doesn't exist
if not os.path.exists(save_path):
    with open(save_path, "w"):
        pass  

exist_set = set()
with open(save_path, "r") as f:
    for line in f:
        data = json.loads(line)
        exist_set.add(data["uid"])

for uid, content in tqdm(user_list):
    start_time = time.time()
    if uid in exist_set:
        continue

    if not content:
        continue

    try:
        service = ChatService(system)
        enhanced = service.ask(content)  # type: str
        enhanced = enhanced.rstrip("\n")

        with open(save_path, "a") as f:
            f.write(json.dumps({"uid": uid, "news": enhanced}) + "\n")
    except Exception as e:
        print(e)

    interval = time.time() - start_time
    if interval <= MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - interval)

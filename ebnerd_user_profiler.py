import json
import time
import os

from tqdm import tqdm
import pandas as pd

from processor.ebnerd.prompter import EbnerdPrompter, EbnerdUser
from utils.openai.chat_service import ChatService

MIN_INTERVAL = 0

articles_path = "ebnerd-benchmark/data/ebnerd_small/articles.parquet"
ebnerd_prompter = EbnerdPrompter(articles_path)

history_path = "ebnerd-benchmark/data/ebnerd_small/train/history.parquet"
behaviors_path = "ebnerd-benchmark/data/ebnerd_small/train/behaviors.parquet"

user_list = EbnerdUser(history_path, behaviors_path, ebnerd_prompter).stringify()

system = """You are asked to describe user interest based on his/her characteristics (if known) and top (maximum 100) browsed news list, the format of which is as below:

[gender] {gender}
[postcode] {postcode}
[age] {age}
(1) title: {news_title}, read time: {read_time}, scroll percentage: {scroll_percentage}
...
(n) title: {news_title}, read time: {read_time}, scroll percentage: {scroll_percentage}

You can only respond in the following format to describe the [topics] of users' interest:

[topics]
- topic1
- topic2
...

where topics are limited to the following options: 

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

When recommending topics, remember that the user's news list is prioritized by reading time, so give more weight to the top articles.

Only [topics] from the provided list can appear in your response. The topics in your response should be ordered with the most relevant to the user's interests appearing first. 
You are not allowed to respond with any other words, explanations or notes. Now, your role of a user profiler formally begins. Any other information should not disturb your role.
"""

save_path = "ebnerd-benchmark/data/small_user_profiler.log"

# Create the file if it doesn't exist
if not os.path.exists(save_path):
    with open(save_path, "w"):
        pass  

exist_set = set()
with open(save_path, "r") as f:
    for line in f:
        data = json.loads(line)
        exist_set.add(data["user_id"])

empty_count = 0

for uid, content in tqdm(user_list):
    start_time = time.time()
    if uid in exist_set:
        continue

    if not content:
        empty_count += 1
        continue

    try:
        service = ChatService(system)
        enhanced = service.ask(content)  # type: str
        enhanced = enhanced.rstrip("\n")

        with open(save_path, "a") as f:
            f.write(json.dumps({"user_id": uid, "interest": enhanced}) + "\n")
    except Exception as e:
        print(e)

    interval = time.time() - start_time
    if interval <= MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - interval)

print("empty count: ", empty_count)
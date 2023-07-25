# TASK2PROMPT = {
#     "ag_news": "What topic best describes this news article?\n{input}\nThe topic is {output}",
#     "ai2_arc": "" ,
#     "amazon_polarity": "{input}\nIs the review positive or negative?\nIt is {output}",
#     "dbpedia_14": ("Pick one category for the following text. The options are - company, "
#       "educational institution, artist, athlete, office holder, mean of transportation, "
#       "building, natural place, village, animal, plant, album, film or written work."
#       "\n{input}\nThe topic of the paragraph is {output}"),
#     # "financial_phrasebank": "{input}\nWhat is the sentiment of the sentence?\nIt is {output}",
#     "financial_phrasebank": "{input} It is {output}",
#     "glue-sst2": "{input}\nIs that sentence positive or negative?\nIt is {output}", 
#     "hellaswag": "",
#     "tweet_eval-offensive": "Is this tweet offensive?\n{input}\n{output}",
#     }

TASK2PROMPT = {"glue-sst2": "{input} My sentiment is {output}",
               "financial_phrasebank": "{input} It is {output}",
               "amazon_polarity": "{input} It is {output}",
               "ag_news": "{input} The topic is {output}",
                "dbpedia_14": "{input} The topic of the paragraph is {output}",
                "tweet_eval-offensive": "{input} Is the sentence hate or non-offensive? {output}",
                }
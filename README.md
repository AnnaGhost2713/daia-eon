# daia-eon

# NLP Methods for CCM (Customer Contact Management)

Congratulations on joining a tremendous learning experiment, and thank you for choosing the E.ON Track! We're glad to invite you into the World of Digital Energy Solutions!

Take your seats and fasten the belts. We are starting... 3 -> 2 -> 1!

# Challenges for E.ON

The data protection law in Germany (DSGVO) states that data containing personally identifiable information (PII) cannot be stored for more than 90 days. In addition, upon customer requests, the data of the customer and everything related to it has to be deleted. This includes data that has been used to train models, the models themselves, the data used for training, and data used for benchmarking performance.
Due to the strict regulations, E.ON faces different challenges. One of them is that models that used data containing PII as training data have to be deleted upon request from the customer, which forces us to retrain our models. Furthermore, data that we use as benchmarks for our models cannot be preserved for more than 90 days.

To utilize the data in a more consistent way and to avoid DSGVO violations, data anonymizing techniques are used.
This includes identifying PII to anonymize (names, telephone numbers, emails, etc.) and replacing the identified PII with random information using frameworks like Faker, LLMs, or similar.

## What classifies as PII?

The data that we want to anonymize is everything that can be used to identify a person. A list of some information that needs to be anonymized is provided:

- Given name
- Last name
- Address
- City
- Contract numbers
- Payments
- Energy consumption
- etc.

If you are not sure, think about the following scenario:
You are a customer at E.ON and sent E.ON an email about some payment-related topic, which could look like this:

```
Hi E.ON Team,
I just got an invoice of 130.5€ for the previous month, but I usually pay 10€.
As a student living on my own, I do not consume that much electricity. I think there is something wrong.
Can you please check my contract with the contract number 98746515?
Thank you.

Regards,
Max Mustermann

```

If someone were to summarize the information provided in the email to you like this:

```
The Customer:
  - Is from Munich
  - He is a student
  - He is living on his own
  - His name is Max Mustermann
  - He has a monthly payment of 10€
  - For June he paid 130.5€
  - His contract number is 98746515

```

What information would you need to anonymize to not identify yourself?
You surely would need to anonymize your name, the city you live in, the contract number, and the payments.
The information on your living situation and your occupation (student) could be a giveaway but are also very hard to anonymize.

As you can see, PII is not defined super precisely, but is basically everything that helps to identify a person. Some information is more obvious (like name, address, contract numbers, etc.) and some are more ambiguous.
Try to find a good trade-off.

# Data Sheet

The dataset originally comprised 181 anonymized customer emails. Some of the emails are removed due to entailed factual information which cannot be disclosed, with 161 emails left.

Every email is placed into a separate text file without any formatting. All files reside under `2025-04-15-golden-dataset`.

The preparation of the masked dataset is part of the task, the anonymized dataset will contain same emails as the emails in the golden dataset but with the masked PII. This dataset should be used to calculate different metrics.

All emails are in German and depict real customer conversations with E.ON Energie Deutschland GmbH.

The data was anonymized manually and replaced with corresponding placeholders.

# Use Cases

The challenge is built around one main NLP task: PII identification for text data.
Either as an addition or as an alternative (depending on the level), the emails provided should be used to generate synthetic emails, which can be used to train a model. The data should therefore be as similar to the original dataset as possible.
For the synthetic data generation, you can make use of Python libraries like [Faker](https://pypi.org/project/Faker/) or similar.

The goal is to create a pipeline to identify PII in emails. Thereby E.ON can utilize the emails for, e.g., training and benchmarking models.

Complexity levels:

- Level 0:
a) Create an LLM-based (e.g., Llama 3.3) setup to anonymize the provided emails using the open-source [Piiranha model](https://huggingface.co/iiiorg/piiranha-v1-detect-personal-information) from HuggingFace.
b) Use the provided emails to generate new synthetic data based on the emails provided.
- Level 1:
a) Calculate the anonymization performance and compare it to the overall performance of the model (as seen on the model website). Give some recommendations on how the model could be improved.
Prepare a split of the provided emails for the test and train subsets (50 emails in the test subset). You need to annotate the date by hand. Use the unified schema for the anonymization like in the following exaple:
    
    ```
    Hallo E.ON,
    
    hiermit Widerrufe ich meinen Vertrag mit der Nummer <<VERTRAGSNUMMER_1>>.
    
    Freundliche Grüße,
    <<VORNAME_1>> <<NACHNAME_1>>
    
    <<STRASSE_HAUSNUMMER_1>>
    <<PLZ_1>> <<ORT_1>>
    
    ```
    

b) Measure the synthetic data performance of the system. How similar/dissimilar are the synthetic emails in comparison to the original ones?

- Level 2:
Implement a second anonymization/synthetic data generation process and compare the performance (metrics, time performance, cost performance, etc.) to the previously developed baseline. What benefits/disadvantages does the new model bring?

The participants are expected to provide a solution at one level at least, starting with Level 0 since the latter tasks are based on the former. It is up to the participants if they want to define a wider or deeper scope for their task.

A "solution" can be a running software solution or a kind of proof-of-concept run manually in the form of a Jupyter Notebook.

# References

Here are some base courses you can have a look at if you are looking for some inspiration:

NLP Courses:

- https://course.spacy.io/en/
- https://huggingface.co/learn/nlp-course/

Tutorials:

- [https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An introduction to explainable AI with Shapley values.html](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html)
- https://towardsdatascience.com/building-sentiment-classifier-using-spacy-3-0-transformers-c744bfc767b
- https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6

Evaluation:

- https://cookbook.openai.com/examples/evaluation/how_to_eval_abstractive_summarization
- https://soletlab.asu.edu/coh-metrix/
- https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0
- https://github.com/mcao516/EntFA
- https://github.com/ThomasScialom/QuestEval
- https://github.com/salesforce/factCC

Frameworks:

- https://ollama.com/
- https://docs.chainlit.io/

Fine-tuning:

- https://www.philschmid.de/fsdp-qlora-llama3
- https://github.com/teticio/llama-squad
- https://www.answer.ai/posts/2024-03-14-fsdp-qlora-deep-dive
- https://arxiv.org/abs/2305.14314

Models:

- https://github.com/jzhang38/TinyLlama
- https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
- https://huggingface.co/models
- https://chat.lmsys.org/?leaderboard

# Our expectations

We expect the participants to stay in touch with the mentors during the whole term. Please do spend time on the tasks and not hope to tackle everything at the last minute.
We are open to your questions and hope to provide as much support as possible to you given your motivation and dedication to the challenge topic.
And you'll experience that even bigger elephants can get swallowed in small pieces by chewing them carefully over a longer time :)
Have fun and happy hacking!

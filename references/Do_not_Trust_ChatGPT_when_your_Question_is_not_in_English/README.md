### Toy-sized test the method and [data](https://github.com/Senyu-Li/LLM-Multilingual-Types) used in the paper "[Don’t Trust ChatGPT when your Question is not in English: A Study of Multilingual Abilities and Types of LLMs](https://aclanthology.org/2023.emnlp-main.491/)" on Korean

Out of the available data (50 instances each in the categories of knowledge, logic, and mathematics), I generated results for half of them, 25 instances each.

##### Knowledge
English answer accuracy: 1.00(25/25), Korean answer accuracy: 0.64(16/25). 
The quality of the questions translated into Korean is good (the questions are in simple formats). But the content of the questions (people, events, etc.) is not familiar in the Korean culture, which may cause errors.
(Does it mean the model does not use the translation into English when asking questions in Korean?)

##### Logic
English answer accuracy: 0.76(19/25), Korean answer accuracy: 0.6(15/25). 
The quality of the questions translated into Korean is sometimes poor. Some of the errors seem to be due to the translation quality.

##### Math
English answer accuracy: 0.92(23/25), Korean answer accuracy: 0.48(12/25). 
The quality of the questions translated into Korean is often poor. Some errors seems to be due to the translation quality and some errors occur during the process.
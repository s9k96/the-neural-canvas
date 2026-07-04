Your assignment would be to pick India's page on Wikipedia in English, Hindi, Telugu, and one more language of your choice. Ask your AI Agent to design a BPE tokenizer in such a way that:

you have 10000 tokens (your vocab) overall for all languages,
(Total English Vocab, say 5000 words)/(Total English tokens) must be around 1.6 or less, let's call this X1
Similarly ratios for your Hingi, Telugu and another language is X2, X3, X4
Sort X1, X2, X3, X4.. say its X4 (largest), X2, X3, X1 (least).
Your assignment score is going to be 1000/(X4 - X1).
What are you submitting:

A Widget that shows these ratios, token statistics and calculations and your self score
Allows me to see your tokenizer (list of all tokens)
URL for this widget on Netlify (or anywhere else you want to host)
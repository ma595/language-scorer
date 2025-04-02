## Code review

A quite reminder of what's going on in this code: 

### Background

BERT (**Bidirectional Encoder Representations from Transformers**)
- A **transformer encoder stack**
- Pretrained on massive text corpora (Wikipedia + BooksCorpus)
- Designed to understand context by **looking in both directions** (left and right of a word)
	- Trained on next sentence prediction
	- masked token prediction
- BERT is a deep learning model that understands the meaning of words **in context**, by looking at **both left and right sides** of a word in a sentence.
	- Unlike earlier models (e.g., GPT-1), BERT looks **both left and right** of a word simultaneously

### My approach to the task

The point of this task is to finetune a LLM for essay scoring. We take an off the shelf model BERT-tiny and fine-tune it on our dataset using HuggingFace's `BertForSequenceClassification`. 

Initially when tackling this, I missed `questions.md` so started investigating it without prompts.
- Created a jupyter notebook that I subsequently deleted containing:
	- distribution of classes:
		- number of tokens. Words != tokens. 
	- A single repetitions
- Impressions of dataset
	- Very clean dataset. Just text and score. No missing values. 

Overall approach:
- Broke it up task into creating a dataset class that inherits from Dataset.
- Then writing training loop code and validation loop code. 
- Could make a `models.py` file to store model definitions. I.e. If we want to do multi-class classification. 


1. Need a Tokeniser.
	- Tokeniser is traditionally packaged with the model you're actually using.
	- In this case I needed to do: 
		- We were not provided with a tokeniser (iirc)
		- So I downloaded one from the internet. 
		- It's bad practice to use a model and tokeniser that are different. Perhaps this was in the task?
		- `tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")`
		- We need to work with numerical data. We will use the Bert tokenizer to convert the text data into numerical data.
		- The word tokenizer splits text into individual words. The Bert tokenizer splits text into subwords. 
    		- This is useful because Bert has a fixed vocabulary size. If a word is not in the vocabulary, it is split into subwords that are in the vocabulary.
			- The data contains @CAPS1 @PERSON - these are special tokens that Bert uses to represent capitalization and named entities. We will need to remove these tokens before tokenizing the data. 

2. Why create a dataset class?
	- Standard interface for dataloader. 
		- Pytorch dataloader expects a dataset to implement `__len__`, `__getitem__`
		- "By defining a Dataset class, I can plug it directly into a DataLoader and handle batching, shuffling, and parallel loading with zero extra work."
	- Handles separation of concerns
		- Dataset does all the processing and data access
		- training code doesn't care where the data comes from or how it's preprocessed.
	- The `__get_item__` magic method, 
		- returns a dict, 'input_ids' : tensor(size=512), 'attention_mask' : tensor(size=512), score : tensor(size=1)
	- In the code I have `__get_item__` processing the data in a lazy way. It might be better to move all this into the constructor so it's processed on object instantiation for small datasets.
	- For large datasets that do not fit into memory - my approach is fine.


3. Why choose `BertForSequenceClassification` rather than a regression head for essay scoring?
	- Essay scores are often treated as ordinal or categorical, and `BertForSequenceClassification` is convenient for classification tasks with discrete labels. It simplifies model setup and leverages cross-entropy loss.
	- All we need to do is set the number of labels to 5 in the model definition. 
		- Under the hood it actually does: `loss = -log(softmax(logits)[label])`
		- loss = CrossEntropyLoss. 
	- Subsequently experimented with floating point scores in `regression-notebook.ipynb`
		- set num_labels=1
		- Use MSELoss
		- Interpret as real valued number.
	- 

4. `BertForSequenceClassification` is a model from the Hugging Face 🤗 Transformers library that wraps a BERT model (e.g., `bert-base-uncased`, `bert-tiny`, etc.) for **sequence-level classification tasks**.
	- It's a model with a classification head on top. 
	- Consists of an encoder
		- processes input tokens and generates contextualised embeddings. 
		- Creates a `[CLS]` embedding which is a summary representation of the entire sequence. This exists at position 0 of the sequence.
		- But each token also gets embedded. 
	- And linear "head" that takes the `[CLS]` embedding and produces logits over the number of classes.
		- `outputs = model(input_ids, attention_mask=mask)`
		- `logits = outputs.logits`,  this is computed from the [CLS] token
		- `nn.Linear(hidden_dim, num_labels) # e.g., Linear(128, 8)`
			- Where the `hidden_dim` is the dimension of the CLS embedding.
	- When we say BERT has a **classification head on top**, we mean that it has an extra **fully connected (linear) layer** that turns the `[CLS]` embedding into predictions over `n` classes. Therefore it can predict 
	- This is a little flawed.
		- It assumes that there is no relationship between the classes. 
		- Therefore a prediction of 4 is just as bad as a prediction of 1. 
		- Model doesn't learn anything w.r.t ordering
		- Predicting **class 1 with 100% confidence** when the truth is 5 gets the **same loss** as predicting **class 4 with 100% confidence** when the truth is 5.
		- MSEloss would be better.
	- Better off with regression or ordinal regression.

5. Python generators
	- In Python, a generator is a special kind of iterator defined using `yield` instead of `return`. It produces items _lazily_, one at a time, which is perfect when data can't all fit into RAM.
	- Generators are super useful in machine learning—especially for handling **large datasets** or **on-the-fly data augmentation**—because they let you **yield one sample at a time**, rather than loading everything into memory. This can massively improve scalability and flexibility.
6. `random_split` is simple but doesn’t preserve class distribution, which can lead to skewed validation performance. Stratified sampling ensures class balance across splits, which is important for imbalanced datasets.

7. Internals of pytorch (DataLoaders, Models, Tensors, Backprop etc)
	1. zero grad → forward → loss → backward → step
	2. pytorch lightning?
	3. Jim's course.

8. Class imbalance
	 - Oversampling minority class
	 - Higher weights to minority class.

9. Evaluation
	- It reports overall accuracy, which can be misleading for imbalanced data — the model may score high accuracy by predicting the majority class. Class-weighted metrics like F1 or per-class accuracy would be more informative.
	- Confusion matrix will give more informative metric
10. Overfitting
	- dropout / regularisation

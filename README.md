# Ollama_privateGPT

#### Inspired from (https://github.com/imartinez/privateGPT) and (https://github.com/jmorganca/ollama)

#### Step 1: Step a Virtual Environment

#### Step 2: Install the Requirements
```
pip install -r requirements.txt
```

#### Step 3: Pull the models (if you already have models loaded in Ollama, then not required)
#### Make sure to have Ollama running on your system from https://ollama.ai
```
ollama run qwen2:0.5b
```

#### Step 4: put your files in the source_documents folder after making a directory
```
mkdir source_documents
```

#### Step 5: Ingest the files (use python3 if on mac)
```
python ingest.py
```

Output should look like this:
```shell
Creating new vectorstore
Loading documents from source_documents
Loading new documents: 100%|██████████████████████| 1/1 [00:01<00:00,  1.99s/it]
Loaded 235 new documents from source_documents
Split into 1268 chunks of text (max. 500 tokens each)
Creating embeddings. May take some minutes...
Ingestion complete! You can now run privateGPT.py to query your documents
```


#### Streamlit UI
```
This project is hosted through streamlit , you can make this work by following:

    - pip install streamlit

And use this command to see the ouptut

    - streamlit run privateGPT.py
```

##### Play with your docs
Enter a query: Define any thing based on source_documents?




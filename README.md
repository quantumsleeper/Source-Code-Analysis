# Source Code Analysis with GenAI

# How to run?
### Steps:

Clone the repository

```bash
git clone https://github.com/quantumsleeper/Source-Code-Analysis.git
```
### 1. Create a conda environment after opening the repository

```bash
conda create -n llmapp python=3.8 -y
```

```bash
conda activate llmapp
```


### 2. Install the requirements
```bash
pip install -r requirements.txt
```

### 3. Create a `.env` file in the root directory and add your OPENAI_API_KEY credentials as follows:

```ini
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


### 4. Run the following command
```bash
python app.py
```

### 5. Launch localhost on your web browser 
```bash
localhost:8080
```


### Techstack Used:

- Python
- LangChain
- Flask
- OpenAI
- GPT 3
- ChoromaDB
import os 
from src.helper import load_embedding, repo_ingestion
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = load_embedding()
persist_directory = "db"

# after store_index.py is run, it'll generate the db 
# we can now load that persisted database from disk and use it normally

vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embeddings)

llm = ChatOpenAI()

memory = ConversationSummaryMemory(
    llm=llm,
    memory_key='chat_history',
    return_messages=True
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm, 
    retriever=vectordb.as_retriever(
        search_type='mmr', search_kwargs={"k": 8}),
    memory=memory
)

## Creating flask route 
# First route will launch web interface 
# for which, we've to render index.html 

@app.route('/',methods=['GET', 'POST'])  # route to display the home page
def index():
    return render_template("index.html")


# ingest the repo 
@app.route('/chatbot', methods=['GET','POST'])
def gitRepo():
    if request.method == 'POST':
        user_input = request.form['question']    # i.e. repo_url
        repo_ingestion(user_input)
        os.system("python store_index.py")

        ## When user will hit 'send' in the web interface, 
        # store_index.py will execute and db is created 

    return jsonify({'response': str(user_input)})


# perform chat operations 
@app.route('/get', methods=['GET','POST'])
def chat():
    msg = request.form['msg']
    input = msg
    print(input)

    if input == 'clear':
        os.system('rm -rf repo')

    result = qa(input)
    print(result['answer'])
    return str(result['answer'])    



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

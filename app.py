from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, ServiceContext, StorageContext, load_index_from_storage
from langchain import OpenAI
import gradio as gr
import dotenv

dotenv.load_dotenv()

def construct_index(directory_path):
    num_outputs = 512
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.3, model_name="text-davinci-003", max_tokens=num_outputs))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    docs = SimpleDirectoryReader(directory_path).load_data()
    aindex = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)
    aindex.storage_context.persist()
    
    return aindex

def chatbot(input_text):
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    bindex = load_index_from_storage(storage_context)
    query_engine = bindex.as_query_engine()
    response = query_engine.query(input_text)
    return response.response

iface = gr.Interface(fn=chatbot,
                     inputs="text",
                     outputs="text",
                     title="AI Chatbot for CA Driver's Instruction Manual")

index = construct_index("docs")
iface.launch(share=True)

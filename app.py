from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, ServiceContext, StorageContext, load_index_from_storage
from langchain import OpenAI
#import gradio as gr
import dotenv
import http.server
import socketserver
import urllib.parse

dotenv.load_dotenv()

PORT = 6006

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

#iface = gr.Interface(fn=chatbot,
#                     inputs="text",
#                     outputs="text",
#                     title="AI Chatbot for CA Driver's Instruction Manual")

index = construct_index("docs")
#iface.launch(share=False)
class CustomRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):

        parsed_url = urllib.parse.urlparse(self.path)
        query_params = urllib.parse.parse_qs(parsed_url.query)
        question = query_params.get('q', [None])[0]

        if question:
            print(question)
            response = chatbot(question)
            print(response)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(response.encode())
            return
        else:
            print("No query parameter 'q' provided")
            self.send_response(400)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            response = "Please provide a query parameter 'q'"
            self.wfile.write(response.encode())
            return

Handler = CustomRequestHandler
httpd = socketserver.TCPServer(('0.0.0.0', PORT), Handler)
print(f"Serving on port {PORT}")

try:
    httpd.serve_forever()
except KeyboardInterrupt:
    print("\nShutting down the server...")
    httpd.shutdown()
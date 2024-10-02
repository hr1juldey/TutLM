import dspy

import time
from tools import*

from typing import List, Optional
from rake_nltk import Rake

class SubQA(dspy.Signature):
    """Break down the main question into smaller questions."""
    Iq = dspy.InputField(desc="The main question to break down.")
    Oq = dspy.OutputField(desc="A well formatted python list of related questions.")

class FindRelevantChunks(dspy.Signature):
    """Find relevant chunks based on a query and graph."""
    query = dspy.InputField(desc="The question to find relevant chunks for.")
    graph = dspy.InputField(desc="The graph containing nodes with embeddings.")
    k = dspy.InputField(default=25)
    relevant_chunks = dspy.OutputField(desc="Top k relevant chunks from the graph.")

class GenerateAnswer(dspy.Signature):
    """Generate an answer based on context with detailed step by step explanation and derivation."""
    context = dspy.InputField(desc="Combined long form context from user and graph.")
    question = dspy.InputField(desc="The original question.")
    answer = dspy.OutputField(desc="The Final answer with detailed explanation. Use proper markdown or latext syntax for formatting.")


class ImgQA(dspy.Signature):
    """Describe the image in the best possible way with detail and accuracy"""
    image=dspy.InputField(desc="can be an image or a path to an image")
    description=dspy.OutputField(desc="detailed description of the image")


class Router(dspy.Signature):
    """You are given a block of markdown text. Your task is to classify each part of the text into one of the following categories General text,Math,Code,Image"""

    text=dspy.InputField(desc="can be any type of markdown or non markdown text")
    general=dspy.OutputField(desc="anything that is not code,math equation or an image or path to an image")
    maths=dspy.OutputField(desc="Mathematical expressions, equations, or symbols that can be enclosed between $...$ or $$...$$ in markdown. or anything that has numbers for calculation")
    code=dspy.OutputField(desc="Programming code blocks or inline code. Code blocks are usually enclosed within triple backticks (```) and inline code with single backticks (`).")
    img=dspy.OutputField(desc="Image links are typically represented with the ![alt text](image-url) markdown syntax or end with ().png) (.jpg) (.jpeg) ")



olm1 = dspy.LM(model="ollama/mistral-nemo:latest", api_base="http://localhost:11434")
olm2=  dspy.LM(model="ollama/mathstral", api_base="http://localhost:11434")
olm3=  dspy.LM(model="ollama/llava:latest", api_base="http://localhost:11434")
olm4= dspy.LM(model="ollama/deepseek-coder-v2:latest", api_base="http://localhost:11434")

dspy.settings.configure(lm=olm1)
# gen=dspy.settings.context(lm=olm1)
# mat=dspy.settings.context(lm=olm2)
# vis=dspy.settings.context(lm=olm3)
# code=dspy.settings.context(lm=olm4)

class GraphRAG(dspy.Module):
    def __init__(self, graph,mode="gen"):
        super().__init__()

        olm1 = dspy.LM(model="ollama/mistral-nemo:latest", api_base="http://localhost:11434")
        olm2=  dspy.LM(model="ollama/mathstral", api_base="http://localhost:11434")
        olm3=  dspy.LM(model="ollama/llava:latest", api_base="http://localhost:11434")
        olm4=  dspy.LM(model="ollama/deepseek-coder-v2:latest", api_base="http://localhost:11434")
        # Step 3: Generate answer based on the selected mode
        if mode == 'gen':
             dspy.settings.context(lm=olm1)
        elif mode == 'mat':
             dspy.settings.context(lm=olm2)
        elif mode == 'vis':
             dspy.settings.context(lm=olm3)
        elif mode == 'code':
             dspy.settings.context(lm=olm4)
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose from 'gen', 'mat', 'vis', or 'code'.")
        
        self.graph = graph
        self.subqa_predictor = dspy.ChainOfThought(SubQA)
        self.find_relevant_chunks_predictor = dspy.ChainOfThought(FindRelevantChunks)
        self.generate_answer_predictor = dspy.ChainOfThought(GenerateAnswer)
        self.mode=mode

    def load_graph(self, filepath):
        """Load the graph from a specified file path using pickle."""
        graph = read_gml(filepath)
        print(f"Graph loaded from {filepath}")
        return graph

    def get_embedding(self, text, model="mxbai-embed-large"):
        """Get embedding for a given text using Ollama API."""
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]

    def calculate_cosine_similarity(self, chunk, query_embedding, embedding):
        """Calculate cosine similarity between a chunk and the query."""
        if np.linalg.norm(query_embedding) == 0 or np.linalg.norm(embedding) == 0:
            return (chunk, 0)  # Handle zero vectors
        cosine_sim = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
        return (chunk, cosine_sim)

    def find_most_relevant_chunks(self, query: str, graph=None, k: int = 25):
        """Find the most relevant chunks based on the query and graph."""
        r = Rake()
        r.extract_keywords_from_text(query)
        keywords = r.get_ranked_phrases()

        relevant_sentences = set()
        for keyword in keywords:
            for node in self.graph.nodes():
                if keyword.lower() in node.lower():
                    relevant_sentences.add(node)

        similarities = {}
        query_embedding = self.get_embedding(query)

        for sentence in relevant_sentences:
            if sentence in self.graph.nodes:
                embedding = self.graph.nodes[sentence].get('embedding')
                if embedding is not None:
                    cosine_sim = self.calculate_cosine_similarity(sentence, query_embedding, embedding)
                    similarities[sentence] = cosine_sim[1]

        sorted_sentences = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
        return sorted_sentences[:k]
    
    
    #dspy.settings.configure(rm=find_most_relevant_chunks)
    
    def answer_query(self, query: str, user_context: Optional[List[str]] = None,mode:str="gen"):
        """Answer a query using the graph and embeddings."""
        
        olm1 = dspy.LM(model="ollama/mistral-nemo:latest", api_base="http://localhost:11434")
        olm2=  dspy.LM(model="ollama/mathstral:latest", api_base="http://localhost:11434")
        olm3=  dspy.LM(model="ollama/llava:latest", api_base="http://localhost:11434")
        olm4=  dspy.LM(model="ollama/deepseek-coder-v2:latest", api_base="http://localhost:11434")

        # Step 3: Generate answer based on the selected mode
        if mode == 'gen':
             dspy.settings.context(lm=olm1)
        elif mode == 'mat':
             dspy.settings.context(lm=olm2)
        elif mode == 'vis':
             dspy.settings.context(lm=olm3)
        elif mode == 'code':
             dspy.settings.context(lm=olm4)
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose from 'gen', 'mat', 'vis', or 'code'.")




        self.mode="gen"
        print(self.mode)
        # Find relevant chunks RAG
        
        relevant_chunks_result = self.find_relevant_chunks_predictor(query=query, graph=self.graph,k=25)
    
        # Join relevant chunks
        graph_context = find_most_relevant_chunks(query=relevant_chunks_result.relevant_chunks, graph=self.graph,k=25)
    
    

        # Combine user context if provided
        context = f"{' '.join(user_context)} {graph_context}" if user_context else graph_context

        time.sleep(.5)
        
        # Generate answer
        self.mode=mode
        print(self.mode)
        
        answer_result = self.generate_answer_predictor(context=context, question=query)
        
        return answer_result.answer


# # Load your graph
# loadpath="/home/riju279/Documents/Code/TutorLM/tmp/bioprocess-engineering-principles-doran3.gml"
# G=load_graph(loadpath)

# graph_rag = GraphRAG(graph=G)






# # Example usage
# Q=[
#     "Calculate the yield of a fermentation process if 10 kg of glucose is consumed and 8 kg of ethanol is produced. What is the yield coefficient?", 

#     "A bioreactor has a volume of 500 L. If the initial concentration of a substrate is 20 g/L and it decreases to 5 g/L after 24 hours, calculate the rate of substrate consumption.",

#     "If a cell culture has a specific growth rate of 0.1 h^-1, how long will it take for the cell concentration to double from an initial concentration of 1 x 10^6 cells/mL?",

#     "Given that the half-life of a drug in the body is 4 hours, calculate how much of a 100 mg dose remains in the body after 12 hours.",

#     "A reaction has an activation energy of 50 kJ/mol. Using the Arrhenius equation, calculate the rate constant at 37°C (310 K) if the rate constant at 25°C (298 K) is known to be 0.1 s^-1.",

#     "If a bioprocess operates at a maximum specific growth rate of 0.2 h^-1 and substrate concentration is limiting, what will be the maximum biomass concentration achievable in a continuous culture?",

#     "Calculate the dilution rate required to maintain a steady state in a continuous stirred-tank reactor (CSTR) with a volume of 1000 L and a desired biomass concentration of 2 g/L if the feed concentration is 10 g/L.",

#     "In an enzyme-catalyzed reaction, if the Km value is 5 mM and the substrate concentration is 15 mM, calculate the reaction velocity if Vmax is known to be 100 µmol/min.",

#     "A pharmaceutical compound has a distribution coefficient (log P) of 3.5. Estimate its permeability across biological membranes using relevant equations.",

#     "If a protein solution has an absorbance of 0.75 at 280 nm, calculate its concentration in mg/mL using the Beer-Lambert law, given that ε (extinction coefficient) is 1.0 mL/(mg·cm).",

#     "A batch bioreactor contains 500 L of culture with an initial cell concentration of 1 x 10^6 cells/mL. After 10 hours, the cell concentration reaches 1 x 10^9 cells/mL. Calculate the specific growth rate and the doubling time.",
    
#     "Given a fermentation process with an initial glucose concentration of 100 g/L, the glucose concentration decreases to 10 g/L after 12 hours. If the product yield is 0.5 g of product per g of glucose, calculate the total product formed at the end of the fermentation.",
    
#     "In a continuous stirred-tank reactor (CSTR), the volumetric flow rate is 0.5 L/h, and the reactor volume is 1000 L. If the substrate concentration in the feed is 50 g/L and the conversion is 90%, calculate the substrate concentration in the reactor.",
    
#     "For a protein with a molecular weight of 50 kDa, calculate the number of moles and molecules in 10 mg of this protein. Assume Avogadro's number to be 6.022 x 10^23 molecules/mol.",
    
#     "A bioreactor is operating at a dilution rate of 0.1 h^-1. If the maximum specific growth rate of the organism is 0.4 h^-1, what will be the steady-state biomass concentration in the reactor if the feed substrate concentration is 100 g/L?",
    
#     "An enzyme with a turnover number (kcat) of 1000 s^-1 and a Km value of 1 mM is acting on a substrate with a concentration of 10 mM. Calculate the initial velocity of the reaction if the enzyme concentration is 0.1 µM.",
    
#     "Calculate the oxygen transfer rate (OTR) in a 2000 L bioreactor if the oxygen concentration in the gas phase is 0.21 mol/L, the gas-liquid mass transfer coefficient (k_La) is 50 h^-1, and the dissolved oxygen concentration in the liquid is 2 mg/L.",
    
#     "A biopharmaceutical drug has a first-order degradation rate constant of 0.03 day^-1 at 25°C. Calculate the shelf life of the drug (time until 90 percent of the drug remains).",
    
#     "In a dialysis process, a protein with a molecular weight of 150 kDa is being separated from smaller molecules. If the diffusion coefficient of the protein is 1 x 10^-11 m^2/s, calculate the time required for the protein to diffuse across a 1 mm membrane.",
    
#     "For a mammalian cell culture operating in a perfusion bioreactor, the perfusion rate is set to 1 reactor volume per day. If the biomass concentration inside the reactor is 10^7 cells/mL, calculate the cell retention efficiency if 5 x 10^6 cells/mL are found in the permeate."
# ]





# J=int(input("enter a number between 1 to 20"))

# main_question_index = J - 1  # Assuming J is defined elsewhere
# main_question = Q[main_question_index]


    
# sub_questions_result = graph_rag.subqa_predictor(Iq=main_question)

# print(sub_questions_result.Oq)

# # Convert Oq string into a proper Python list
# sub_questions =json.loads(sub_questions_result.Oq)                  # [question.strip('- ').strip() for question in sub_questions_result.Oq.split('-') if question]

# sub_questions




# for sub_question in sub_questions:
#     thought_process_result = graph_rag.answer_query(query=sub_question,mode="gen")
#     time.sleep(.5)
    
# # Final answer generation
# final_answer_result = graph_rag.answer_query(query=main_question, user_context=[thought_process_result],mode="mat")

# print(f"---- \n\n The question is: \n\n {main_question} \n\n Final Answer is \n \n {final_answer_result} \n\n --------")

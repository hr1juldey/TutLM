from Config  import*





## this fucntion below reads files


def read_file(file_path):
    """Read the content of a Markdown or PDF file."""
    if file_path.endswith('.pdf'):
        text=process_pdf(file_path, save=False)
        return text

    elif file_path.endswith('.md') or file_path.endswith('.markdown') or file_path.endswith(".txt") or file_path.endswith(".py"):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    else:
        raise ValueError("Unsupported file type. Please provide a Markdown or PDF file.")

# this function was intended for chapter finding but could not use it due to complexity


def detect_table_of_contents(file):
    """Detect the Table of Contents in the document."""
    # Sample TOCEntry class definition
    class TOCEntry:
        def __init__(self, level, title, page):
            self.level = level
            self.title = title
            self.page = page
    
    # Function to extract chapter titles from a PDF file
    def get_chapter_titles_from_pdf(filename):
        # Extract TOC entries using the provided filename
        toc_entries = extract_toc(filename)
        
        # Extracting chapter titles using a list comprehension
        chapter_titles = [entry.title for entry in toc_entries]
        
        return chapter_titles
    
    titles_list=get_chapter_titles_from_pdf(file)
    print(f"\n\n titles list {titles_list} \n\n \n\n")
    return titles_list


def split_text_into_sections(file):
    """Split text into chapters, pages, paragraphs, sentences, and words."""
    

    text=read_file(file)

    def split_text(text, delimiters):
        """Split text using multiple delimiters."""
        # Create a regex pattern that matches any of the delimiters
        pattern = '|'.join(map(re.escape, delimiters))
        return re.split(pattern, text)

    chapternames =detect_table_of_contents(file)  # List of chapters already given for making it fast
    

    chapters = split_text(text,chapternames) # deactivate if not using the Biochem.md or rb.md
    #chapters=text.split('----')  # activate if not using the Biochem.md

    graph = nx.Graph()
    stop_words = set(stopwords.words('english'))  # Load English stopwords

    

    def process_chapter(chapter):
        """Process a single chapter into pages, paragraphs, sentences, and words."""
        start=time.time()
        pattern = r'(-{3,4}|\n\n)'
        
        pages = chapter.split(pattern)  # Assuming pages are separated by double newlines

        for page in pages:
            paragraphs = re.split(r'\n+', page)  # Split into paragraphs
            

            for paragraph in paragraphs:
                #print(paragraph) # Paragraph
                sentences = sent_tokenize(paragraph)  # Split into sentences using NLTK
                for sentence in sentences:
                    words = word_tokenize(sentence)  # Split into words using NLTK
                    filtered_words = [word for word in words if word.lower() not in stop_words]  # Remove stopwords
                    
                    # Create nodes in the graph
                    graph.add_node(sentence)
                    sentence_embedding = get_embedding(sentence)
                    graph.nodes[sentence]['embedding'] = sentence_embedding  # Store embedding in the graph
                    
                    for word in filtered_words:
                        graph.add_node(word)
                        graph.add_edge(sentence, word)  # Connect sentence to its words

                    # Extract keywords using RAKE
                    r = Rake()
                    r.extract_keywords_from_text(sentence)
                    keywords = r.get_ranked_phrases()
                    graph.nodes[sentence]['keywords'] = keywords  # Store keywords in the graph
                    for keyword in keywords:
                        graph.add_node(keyword)
                        keyword_embedding = get_embedding(keyword)
                        graph.nodes[keyword]['embedding'] = keyword_embedding  # Store embedding in the graph
                        graph.add_edge(sentence, keyword)  # Connect sentence to its keywords
                        
                graph.add_edge(page, paragraph)  # Connect page to its paragraphs
            graph.add_edge(chapter, page)  # Connect chapter to its pages
        
        end=time.time()
        looptime=(end-start)
        print(f"------------- \n\n chapter completed in {looptime} \n\n -------------")

    # Use multithreading to process chapters
    cpu_count = os.cpu_count()
    max_threads=cpu_count*30
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(process_chapter, chapter) for chapter in chapters]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # Wait for the chapter processing to complete
                
            except Exception as e:
                print(f"Error processing chapter: {e}")

    return graph


def process_pdfs_in_folder(folder_path, save):
    #graphs = []  # List to hold the resulting graphs
    saves = []

    # Progress bar for processing PDFs
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]  # Filter out only PDFs
    for filename in tqdm(pdf_files, desc="Processing PDFs"):
        full_path = os.path.join(folder_path, filename)  # Get full file path
        graph = split_text_into_sections(full_path)  # Process the PDF
        print(f"\n\n ------------- \n\n Graph creation complete for {filename} \n\n -------------- \n\n ")
        
        # Remove the '.pdf' extension and save as .gml
        filename_no_ext = os.path.splitext(filename)[0]  # Get the file name without the extension
        savepath = os.path.join(save, f"{filename_no_ext}.gml")  # Save as .gml
        save_graph(graph, savepath)
        saves.append(savepath)  # Store the exact savepath

    msg=print(f"all books processed, saved at {saves}")

    return msg


# If there is a library of books




# GraphRAG takes a lot of time to calculate on big books so we will save the graphs as yaml

def save_graph(graph, filepath):
    """Save the graph to a specified file path using pickle."""
    save_gml(graph,filepath)
    print(f"Graph saved to {filepath} as gml file")

def load_graph(filepath):
    """Load the graph from a specified file path using pickle."""
    graph=read_gml(filepath)
    print(f"Graph loaded from {filepath}")
    return graph



# The embedding Function

def get_embedding(text, model="mxbai-embed-large"):
    """Get embedding for a given text using Ollama API."""
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

# This function below gets the similarity of keywords in question with the huge text

def calculate_cosine_similarity(chunk, query_embedding, embedding):
    """Calculate cosine similarity between a chunk and the query."""
    if np.linalg.norm(query_embedding) == 0 or np.linalg.norm(embedding) == 0:
        return (chunk, 0)  # Handle zero vectors
    cosine_sim = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
    return (chunk, cosine_sim)



# The Retrival portion of the graphrag

def find_most_relevant_chunks(query, graph,k=25):
    """Find the most relevant chunks based on the graph and cosine similarity to the query."""
    # Step 1: Extract keywords from the query using RAKE
    r = Rake()
    r.extract_keywords_from_text(query)
    keywords = r.get_ranked_phrases()

    # Step 2: Find relevant sentences in the graph based on keywords
    relevant_sentences = set()
    for keyword in keywords:
        for node in graph.nodes():
            if keyword.lower() in node.lower():  # Check if keyword is in the node
                relevant_sentences.add(node)  # Add the whole sentence

    # Step 3: Calculate embeddings for relevant sentences
    similarities = {}
    query_embedding = get_embedding(query)

    for sentence in relevant_sentences:
        if sentence in graph.nodes:
            embedding = graph.nodes[sentence].get('embedding')
            if embedding is not None:
                cosine_sim = calculate_cosine_similarity(sentence, query_embedding, embedding)
                similarities[sentence] = cosine_sim[1]  # Store only the similarity score

    # Sort sentences by similarity
    sorted_sentences = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    return sorted_sentences[:k]  # Return top k relevant sentences





def answer_query(query: str, graph, Ucontext: Optional[List[str]] = None,model='mistral-nemo:latest', k: int = 25):
    """Answer a query using the graph and embeddings, combining both system context and user-provided context."""
    
    # Find the most relevant chunks from the graph
    relevant_chunks = find_most_relevant_chunks(query, graph, k=k)
    
    # Join the relevant chunks from the graph into a single string
    graph_context = " ".join(chunk for chunk, _ in relevant_chunks)
    
    # If Ucontext is provided, join it with the graph context
    if Ucontext:
        user_context = " ".join(Ucontext)
        context = f"{user_context} {graph_context}"  # Combine user context and graph context
    else:
        context = graph_context
    
    # Generate the response using the combined context
    response = ollama.generate(model=model, prompt=f"based on the Context: {context} answer the Question: {query}")
    
    # Return the response if available, otherwise return a default message
    return response.get('response', "No answer generated.")


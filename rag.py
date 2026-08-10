import os
import faiss
import numpy as np

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found.\n"
        "Please add your Gemini API key to the .env file."
    )


# ============================================================
# 2. CREATE GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=API_KEY
)


# ============================================================
# 3. KNOWLEDGE BASE
# ============================================================
# No external document is required.
# The information is stored directly in this Python file.

knowledge_base = [

    """
    Artificial Intelligence (AI) is a field of computer science
    that focuses on creating systems that can perform tasks that
    normally require human intelligence. These tasks include
    learning, reasoning, problem solving, understanding language,
    and recognizing patterns.
    """,

    """
    Machine Learning (ML) is a subset of Artificial Intelligence.
    Machine Learning allows computers to learn patterns from data
    without being explicitly programmed for every individual task.
    The major types of Machine Learning are supervised learning,
    unsupervised learning, and reinforcement learning.
    """,

    """
    Deep Learning is a subset of Machine Learning that uses
    artificial neural networks containing multiple layers.
    Deep Learning is commonly used for image recognition,
    speech recognition, natural language processing, and
    autonomous systems.
    """,

    """
    Cybersecurity is the practice of protecting computers,
    networks, applications, systems, and data from unauthorized
    access, attacks, damage, and theft. Common cybersecurity
    techniques include encryption, authentication, firewalls,
    intrusion detection, vulnerability assessment, and access
    control.
    """,

    """
    A firewall is a security system that monitors and controls
    incoming and outgoing network traffic according to predefined
    security rules. Firewalls can help prevent unauthorized network
    access and can be implemented as hardware, software, or cloud
    services.
    """,

    """
    Encryption is a cybersecurity technique that converts readable
    information called plaintext into an encoded form called
    ciphertext. A cryptographic key is required to decrypt the
    ciphertext and recover the original information.
    """,

    """
    Retrieval-Augmented Generation, commonly called RAG, combines
    information retrieval with generative Artificial Intelligence.
    A RAG system first retrieves relevant information from a
    knowledge base and then provides that information to a language
    model as context for generating an answer.
    """,

    """
    RAG can help reduce hallucinations because the language model
    receives relevant information from an external knowledge base
    before generating its answer. The quality of the answer depends
    on the quality of the retrieved information.
    """,

    """
    Python is a high-level programming language commonly used for
    web development, automation, data science, machine learning,
    artificial intelligence, and cybersecurity. Python is popular
    because its syntax is relatively simple and it has a large
    ecosystem of libraries.
    """,

    """
    SQL stands for Structured Query Language. SQL is used to
    communicate with relational databases. Common SQL operations
    include SELECT, INSERT, UPDATE, and DELETE. SELECT is commonly
    used to retrieve information from a database.
    """
]


# ============================================================
# 4. LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# 5. INDEXING
# ============================================================

def create_index():

    print("\n==============================================")
    print("INDEXING")
    print("==============================================")

    print(
        f"\nNumber of knowledge chunks: "
        f"{len(knowledge_base)}"
    )

    # Convert knowledge into embeddings
    embeddings = embedding_model.encode(
        knowledge_base,
        convert_to_numpy=True
    )

    # Convert to float32 for FAISS
    embeddings = embeddings.astype(
        "float32"
    )

    # Get embedding dimension
    dimension = embeddings.shape[1]

    print(
        f"Embedding dimension: {dimension}"
    )

    # Create FAISS index
    index = faiss.IndexFlatL2(
        dimension
    )

    # Add embeddings to FAISS
    index.add(
        embeddings
    )

    print(
        f"Indexed {index.ntotal} knowledge chunks."
    )

    return index


# ============================================================
# 6. RETRIEVAL
# ============================================================

def retrieve_documents(
    question,
    index,
    top_k=3
):

    # Convert question into embedding
    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    question_embedding = question_embedding.astype(
        "float32"
    )

    # Search FAISS
    distances, indices = index.search(
        question_embedding,
        top_k
    )

    retrieved_chunks = []

    for position, index_number in enumerate(
        indices[0]
    ):

        if index_number < len(
            knowledge_base
        ):

            chunk = knowledge_base[
                index_number
            ]

            retrieved_chunks.append(
                chunk.strip()
            )

    return retrieved_chunks, distances[0]


# ============================================================
# 7. RESPONSE GENERATION
# ============================================================

def generate_answer(
    question,
    retrieved_chunks
):

    # Combine retrieved information
    context = "\n\n".join(
        retrieved_chunks
    )

    prompt = f"""
You are a Retrieval-Augmented Generation (RAG)
question-answering assistant.

Answer the user's question using ONLY the
provided context.

CONTEXT:
{context}

USER QUESTION:
{question}

Rules:

1. Use only information from the context.
2. Do not invent information.
3. If the context does not contain enough information,
   say that the information is not available in the
   knowledge base.
4. Give a simple and clear answer.
5. Keep the answer concise.
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return response.text.strip()


# ============================================================
# 8. MAIN PROGRAM
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("        RAG-BASED QUESTION ANSWERING SYSTEM")
    print("=" * 60)

    print("\nLLM Model:")
    print(MODEL)

    # --------------------------------------------------------
    # INDEXING
    # --------------------------------------------------------

    index = create_index()

    print("\nIndexing completed successfully.")

    # --------------------------------------------------------
    # INTERACTIVE QUESTION LOOP
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("INTERACTIVE QUESTION ANSWERING")
    print("=" * 60)

    print(
        "\nYou can ask questions about:"
    )

    print(
        "- Artificial Intelligence"
    )

    print(
        "- Machine Learning"
    )

    print(
        "- Deep Learning"
    )

    print(
        "- Cybersecurity"
    )

    print(
        "- Firewalls"
    )

    print(
        "- Encryption"
    )

    print(
        "- RAG"
    )

    print(
        "- Python"
    )

    print(
        "- SQL"
    )

    print(
        "\nType 'exit' to stop the program."
    )

    # --------------------------------------------------------
    # CONTINUOUS QUESTIONS
    # --------------------------------------------------------

    while True:

        print("\n" + "-" * 60)

        question = input(
            "Enter your question: "
        ).strip()

        # Exit condition
        if question.lower() in [
            "exit",
            "quit",
            "q"
        ]:

            print(
                "\nThank you for using the RAG system!"
            )

            break

        # Empty question
        if not question:

            print(
                "Please enter a question."
            )

            continue

        # ----------------------------------------------------
        # RETRIEVAL
        # ----------------------------------------------------

        print("\nRetrieving relevant information...")

        retrieved_chunks, distances = retrieve_documents(
            question,
            index,
            top_k=3
        )

        print(
            f"Retrieved {len(retrieved_chunks)} "
            f"relevant chunks."
        )

        # ----------------------------------------------------
        # DISPLAY RETRIEVED INFORMATION
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("RETRIEVED CONTEXT")
        print("=" * 60)

        for i, chunk in enumerate(
            retrieved_chunks,
            start=1
        ):

            print(
                f"\n--- Chunk {i} ---"
            )

            print(chunk)

            print(
                f"\nSimilarity distance: "
                f"{distances[i - 1]:.4f}"
            )

        # ----------------------------------------------------
        # GENERATE ANSWER
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("GENERATING RESPONSE")
        print("=" * 60)

        try:

            answer = generate_answer(
                question,
                retrieved_chunks
            )

            # ------------------------------------------------
            # FINAL ANSWER
            # ------------------------------------------------

            print("\n" + "=" * 60)
            print("FINAL ANSWER")
            print("=" * 60)

            print("\n" + answer)

        except Exception as e:

            print(
                "\nError while generating response:"
            )

            print(e)

            print(
                "\nCheck your Gemini API key, "
                "model name, and API access."
            )


# ============================================================
# 9. START PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
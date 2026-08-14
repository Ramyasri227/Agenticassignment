import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3-flash-preview"
)

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found.\n"
        "Please check your .env file."
    )


# ============================================================
# 2. CREATE SQLITE DATABASE
# ============================================================

DATABASE_URL = "sqlite:///college.db"

engine = create_engine(DATABASE_URL)


# ============================================================
# 3. CREATE STUDENTS TABLE
# ============================================================

with engine.begin() as connection:

    connection.execute(text("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            marks INTEGER NOT NULL
        )
    """))


# ============================================================
# 4. INSERT SAMPLE DATA
# ============================================================

with engine.begin() as connection:

    # Clear previous records
    connection.execute(
        text("DELETE FROM students")
    )

    # Insert sample students
    connection.execute(text("""
        INSERT INTO students
        (id, name, department, marks)
        VALUES
        (1, 'Ramya', 'CSE', 92),
        (2, 'Anu', 'CSE', 85),
        (3, 'Ravi', 'ECE', 76),
        (4, 'Priya', 'CSE', 90),
        (5, 'Kiran', 'ECE', 68),
        (6, 'Rahul', 'CSE', 78),
        (7, 'Sneha', 'ECE', 88),
        (8, 'Arjun', 'CSE', 95)
    """))


print("\nDatabase created successfully!")


# ============================================================
# 5. CONNECT LANGCHAIN TO DATABASE
# ============================================================

db = SQLDatabase(engine)

print("\nAvailable database tables:")

for table in db.get_usable_table_names():
    print("-", table)


# ============================================================
# 6. CREATE GEMINI MODEL
# ============================================================

print("\nUsing Gemini model:", MODEL)

llm = ChatGoogleGenerativeAI(
    model=MODEL,
    google_api_key=API_KEY,
    temperature=0
)


# ============================================================
# 7. CREATE SQL AGENT
# ============================================================

agent = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="tool-calling",
    verbose=True,
    handle_parsing_errors=True
)


# ============================================================
# 8. FORMAT AGENT RESPONSE
# ============================================================

def format_output(output):

    """
    Gemini may return the final answer as either:

    1. A normal string
    2. A list containing dictionaries such as:
       [{'type': 'text', 'text': '...'}]

    This function extracts only the useful text.
    """

    # Case 1: Normal string
    if isinstance(output, str):
        return output

    # Case 2: List of content blocks
    if isinstance(output, list):

        text_parts = []

        for item in output:

            if isinstance(item, dict):

                # Extract text from Gemini content block
                if item.get("type") == "text":

                    text_value = item.get("text", "")

                    if text_value:
                        text_parts.append(text_value)

            elif isinstance(item, str):

                text_parts.append(item)

        return "\n".join(text_parts)

    # Case 3: Anything else
    return str(output)


# ============================================================
# 9. ASK QUESTION FUNCTION
# ============================================================

def ask_question(question):

    print("\n")
    print("=" * 70)

    print("USER QUESTION:")
    print(question)

    print("=" * 70)

    try:

        # Send question to SQL Agent
        response = agent.invoke(
            {
                "input": question
            }
        )

        # Get agent output
        output = response.get("output", "")

        # Clean the output
        final_answer = format_output(output)

        print("\nFINAL ANSWER:")
        print(final_answer)

    except Exception as error:

        print("\nERROR:")
        print(error)

    print("=" * 70)


# ============================================================
# 10. START SQL AGENT
# ============================================================

print("\n")
print("=" * 70)
print("              SQL AGENT WITH TOOL USE")
print("=" * 70)

print("""
The agent can answer questions about the student database.

Example questions:

1. Which student has the highest marks?
2. Show all CSE students.
3. What is the average mark of CSE students?
4. How many ECE students are there?
5. Who scored more than 85?
6. Show the top 3 students.
7. Which department has the highest average marks?
8. Show the students in descending order of marks.
9. What is the information of student with ID 6?

Type 'exit' to stop.
""")

print("=" * 70)


# ============================================================
# 11. INTERACTIVE LOOP
# ============================================================

while True:

    question = input("\nYou: ")

    # Exit program
    if question.strip().lower() == "exit":

        print("\nSQL Agent stopped.")
        break

    # Ignore empty input
    if not question.strip():

        print("Please enter a question.")
        continue

    # Ask the agent
    ask_question(question)
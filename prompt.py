import os
from dotenv import load_dotenv
from google import genai


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in .env file."
    )


# ============================================================
# 2. CREATE GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=API_KEY
)


# ============================================================
# 3. FUNCTION TO CALL GEMINI
# ============================================================

def ask_llm(prompt):

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return response.text.strip()


# ============================================================
# 4. STEP 1 - SUMMARIZATION
# ============================================================

def generate_summary(text):

    prompt = f"""
You are an expert summarization assistant.

Summarize the following text in simple language.

TEXT:
{text}

Requirements:
- Keep the important information.
- Remove unnecessary details.
- Do not add information that is not present.
- Keep the summary between 3 and 5 sentences.
"""

    return ask_llm(prompt)


# ============================================================
# 5. STEP 2 - KEY POINT EXTRACTION
# ============================================================

def extract_key_points(summary):

    prompt = f"""
You are an information extraction assistant.

Extract the most important points from the summary below.

SUMMARY:
{summary}

Requirements:
- Provide exactly 5 key points.
- Use simple language.
- Each point should be one sentence.
- Do not introduce new information.
"""

    return ask_llm(prompt)


# ============================================================
# 6. STEP 3 - QUESTION GENERATION
# ============================================================

def generate_questions(key_points):

    prompt = f"""
You are an educational question-generation assistant.

Based on the following key points, create 3 useful questions.

KEY POINTS:
{key_points}

Requirements:
- Generate exactly 3 questions.
- Questions should test understanding.
- Questions should be directly related to the key points.
- Do not provide answers.
"""

    return ask_llm(prompt)


# ============================================================
# 7. STEP 4 - FINAL STRUCTURED OUTPUT
# ============================================================

def generate_final_output(
    summary,
    key_points,
    questions
):

    prompt = f"""
Organize the following information into a clean final report.

SUMMARY:
{summary}

KEY POINTS:
{key_points}

QUESTIONS:
{questions}

Format the response exactly as:

SUMMARY:
<summary>

KEY POINTS:
<key points>

QUESTIONS:
<questions>
"""

    return ask_llm(prompt)


# ============================================================
# 8. MAIN PROGRAM
# ============================================================

def main():

    print("=" * 60)
    print("       PROMPT CHAINING FOR SUMMARIZATION")
    print("=" * 60)

    print("\nUsing model:")
    print(MODEL)

    print("\nEnter the text you want to process.")
    print("Type 'END' on a new line when finished.\n")

    # --------------------------------------------------------
    # GET MULTI-LINE INPUT
    # --------------------------------------------------------

    lines = []

    while True:

        line = input()

        if line.strip().upper() == "END":
            break

        lines.append(line)

    text = "\n".join(lines).strip()

    if not text:

        print("\nNo text was entered.")
        return

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("STEP 1: SUMMARIZATION")
    print("=" * 60)

    print("\nGenerating summary...")

    summary = generate_summary(text)

    print("\nSummary:")
    print(summary)

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("STEP 2: KEY POINT EXTRACTION")
    print("=" * 60)

    print("\nExtracting key points...")

    key_points = extract_key_points(
        summary
    )

    print("\nKey Points:")
    print(key_points)

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("STEP 3: QUESTION GENERATION")
    print("=" * 60)

    print("\nGenerating questions...")

    questions = generate_questions(
        key_points
    )

    print("\nQuestions:")
    print(questions)

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("STEP 4: FINAL OUTPUT")
    print("=" * 60)

    print("\nCreating final report...")

    final_output = generate_final_output(
        summary,
        key_points,
        questions
    )

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print("\n" + final_output)


# ============================================================
# 9. RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
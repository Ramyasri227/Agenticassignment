import sqlite3
import os
import pandas as pd
from dotenv import load_dotenv
from google import genai


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found.\n"
        "Please add it to your .env file."
    )


# ============================================================
# 2. CREATE GEMINI CLIENT
# ============================================================

client = genai.Client(api_key=API_KEY)


# ============================================================
# 3. DATABASE CONFIGURATION
# ============================================================

DB_NAME = "company.db"


# ============================================================
# 4. CREATE DATABASE
# ============================================================

def create_database():

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Delete existing table
    cursor.execute("DROP TABLE IF EXISTS employees")

    # Create employees table
    cursor.execute("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT,
            department TEXT,
            salary INTEGER,
            experience INTEGER,
            location TEXT
        )
    """)

    # Employee data
    employees = [
        (1, "Anjali", "IT", 65000, 3, "Hyderabad"),
        (2, "Rahul", "HR", 45000, 2, "Chennai"),
        (3, "Priya", "IT", 80000, 5, "Bangalore"),
        (4, "Arjun", "Finance", 70000, 4, "Hyderabad"),
        (5, "Sneha", "IT", 90000, 6, "Mumbai"),
        (6, "Vikram", "HR", 50000, 3, "Delhi"),
        (7, "Kiran", "Finance", 85000, 7, "Bangalore"),
        (8, "Meena", "IT", 75000, 4, "Chennai")
    ]

    cursor.executemany("""
        INSERT INTO employees
        (id, name, department, salary, experience, location)
        VALUES (?, ?, ?, ?, ?, ?)
    """, employees)

    conn.commit()

    return conn


# ============================================================
# 5. RETRIEVE DATABASE SCHEMA
# ============================================================

def get_database_schema(conn):

    cursor = conn.cursor()

    cursor.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        AND name NOT LIKE 'sqlite_%'
    """)

    tables = cursor.fetchall()

    schema = ""

    for table in tables:

        table_name = table[0]

        schema += f"\nTable: {table_name}\n"

        cursor.execute(
            f"PRAGMA table_info({table_name})"
        )

        columns = cursor.fetchall()

        for column in columns:

            column_name = column[1]
            column_type = column[2]

            schema += f"- {column_name}: {column_type}\n"

    return schema


# ============================================================
# 6. GENERATE SQL USING GEMINI
# ============================================================

def generate_sql(question, schema):

    prompt = f"""
You are an expert Text-to-SQL system.

Convert the user's natural-language question into a valid
SQLite SQL query.

DATABASE SCHEMA:
{schema}

USER QUESTION:
{question}

IMPORTANT RULES:

1. Generate ONLY SQL.
2. Do NOT use markdown.
3. Do NOT explain the query.
4. Use ONLY the tables and columns provided in the schema.
5. Generate SELECT queries only.
6. Never use INSERT.
7. Never use UPDATE.
8. Never use DELETE.
9. Never use DROP.
10. Never use ALTER.
11. Never use CREATE.
12. Do not modify the database.

Return only the SQL query.
"""

    try:

        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )

        sql_query = response.text.strip()

        # Remove markdown if model returns it
        sql_query = sql_query.replace("```sql", "")
        sql_query = sql_query.replace("```", "")

        return sql_query.strip()

    except Exception as e:

        print("\nERROR WHILE GENERATING SQL:")
        print(e)

        return None


# ============================================================
# 7. VALIDATE SQL
# ============================================================

def validate_sql(sql_query):

    if not sql_query:
        return False

    sql_upper = sql_query.upper().strip()

    # Must start with SELECT
    if not sql_upper.startswith("SELECT"):
        return False

    dangerous_commands = [
        "INSERT ",
        "UPDATE ",
        "DELETE ",
        "DROP ",
        "ALTER ",
        "CREATE ",
        "REPLACE ",
        "TRUNCATE ",
        "ATTACH ",
        "DETACH "
    ]

    for command in dangerous_commands:

        if command in sql_upper:
            return False

    return True


# ============================================================
# 8. EXECUTE SQL
# ============================================================

def execute_sql(conn, sql_query):

    try:

        result = pd.read_sql_query(
            sql_query,
            conn
        )

        return result

    except Exception as e:

        print("\nSQL EXECUTION ERROR:")
        print(e)

        return None


# ============================================================
# 9. GENERATE FINAL HUMAN-FRIENDLY ANSWER
# ============================================================

def generate_final_answer(question, sql_query, result):

    result_text = result.to_string(index=False)

    prompt = f"""
You are a helpful data analyst.

USER QUESTION:
{question}

SQL QUERY:
{sql_query}

DATABASE RESULT:
{result_text}

Give the user a short and clear answer based ONLY
on the database result.

Do not mention SQL.
Do not mention Python.
Do not mention the database.
Do not invent information.
"""

    try:

        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )

        return response.text.strip()

    except Exception as e:

        print("\nERROR WHILE GENERATING FINAL ANSWER:")
        print(e)

        return None


# ============================================================
# 10. MAIN PROGRAM
# ============================================================

def main():

    print("=" * 55)
    print("              TEXT-TO-SQL SYSTEM")
    print("=" * 55)

    print("\nUsing Gemini model:")
    print(MODEL)

    # Create database
    conn = create_database()

    # Retrieve schema
    schema = get_database_schema(conn)

    print("\nRetrieved Database Schema:")
    print(schema)

    # Get question
    question = input(
        "\nEnter your question: "
    ).strip()

    if not question:

        print("\nPlease enter a question.")

        conn.close()
        return

    # Generate SQL
    print("\nGenerating SQL...")

    sql_query = generate_sql(
        question,
        schema
    )

    if not sql_query:

        print("\nCould not generate SQL.")

        conn.close()
        return

    # Display SQL
    print("\n" + "=" * 55)
    print("GENERATED SQL")
    print("=" * 55)

    print(sql_query)

    # Validate SQL
    if not validate_sql(sql_query):

        print("\nUnsafe or invalid SQL query.")
        print("Only SELECT queries are allowed.")

        conn.close()
        return

    # Execute SQL
    print("\nExecuting SQL...")

    result = execute_sql(
        conn,
        sql_query
    )

    if result is None:

        conn.close()
        return

    # Display result
    print("\n" + "=" * 55)
    print("QUERY RESULT")
    print("=" * 55)

    if result.empty:

        print("No records found.")

    else:

        print(
            result.to_string(index=False)
        )

    # Generate final answer
    if not result.empty:

        print("\nGenerating final answer...")

        final_answer = generate_final_answer(
            question,
            sql_query,
            result
        )

        print("\n" + "=" * 55)
        print("FINAL ANSWER")
        print("=" * 55)

        if final_answer:

            print(final_answer)

        else:

            print("Could not generate final answer.")

    # Close database
    conn.close()

    print("\n" + "=" * 55)
    print("Program completed.")
    print("=" * 55)


# ============================================================
# 11. RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
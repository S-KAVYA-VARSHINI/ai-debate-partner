
import time

DEBATE_SYSTEM_PROMPT = """
You are an AI Debate Partner.

Your job is to debate against the user's position.

Rules:
1. Understand the user's actual claim before responding.
2. Take the opposing side.
3. Directly address the user's latest argument.
4. Give a logical and meaningful counterargument.
5. Do not simply say "I disagree."
6. Do not repeat the same argument unnecessarily.
7. If the user makes a strong point, acknowledge it briefly before challenging it.
8. Ask a challenging follow-up question when appropriate.
9. Stay respectful and professional.
10. Do not invent statistics, studies, or sources.
11. Maintain context from the previous conversation.
12. The user can debate about any topic.
"""

def debate_with_rag(
    client,
    retrieve_knowledge,
    user_argument,
    history=None,
    retries=3
):
    if history is None:
        history = []

    retrieved = retrieve_knowledge(user_argument, k=2)

    knowledge_context = "\n\n".join(retrieved)

    conversation = ""

    for message in history:
        conversation += f"{message['role']}: {message['content']}\n"

    conversation += f"User: {user_argument}\nAI:"

    prompt = f"""
{DEBATE_SYSTEM_PROMPT}

Relevant knowledge retrieved from the knowledge base:
----------------
{knowledge_context}
----------------

Previous conversation:
----------------
{conversation}
----------------

Instructions:
- Debate against the user's position.
- Directly respond to their latest argument.
- Use retrieved knowledge when relevant.
- Do not mention RAG or the knowledge base.
- Do not blindly agree with the user.
- Do not invent facts or statistics.
- Keep the response clear and suitable for a live debate.

Now provide your counterargument.
"""

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )

            return response.text, retrieved

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5)
            else:
                raise e

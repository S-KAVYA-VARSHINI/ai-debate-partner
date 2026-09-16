
def evaluate_debate(client, history):

    conversation = ""

    for message in history:
        conversation += (
            f"{message['role']}: "
            f"{message['content']}\n"
        )

    prompt = f"""
You are a debate evaluator.

Evaluate ONLY the USER's performance.

Debate:
{conversation}

Give scores from 1 to 10 for:

Reasoning
Evidence
Rebuttal
Clarity
Consistency
Communication

Then provide:

Overall score
Two strengths
Two areas for improvement
One logical fallacy if present

Use this format:

Reasoning: X/10
Evidence: X/10
Rebuttal: X/10
Clarity: X/10
Consistency: X/10
Communication: X/10
Overall: X/10

Strengths:
- ...
- ...

Areas for improvement:
- ...
- ...

Logical fallacy:
...
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text

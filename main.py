from openai import OpenAI
from pydantic import BaseModel, Field

# Initialize the OpenAI client (automatically reads your OPENAI_API_KEY)
client = OpenAI()


# Define your strict Pydantic output structure
class UserProfile(BaseModel):
    name: str
    role: str
    skills: list[str]
    years_experience: int = Field(ge=0)


# Call OpenAI using Structured Outputs
completion = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "Extract user profile details accurately from the provided text.",
        },
        {
            "role": "user",
            "content": "My name is Irvin Ramos Castillo, I work as an AI Engineer with 2 years of experience. My primary technical stack includes Python, Pydantic, MASM, and C++.",
        },
    ],
    response_format=UserProfile,
)

# Access the validated, parsed data object directly from the model
user = completion.choices.message.parsed

print("--- Live OpenAI API & Pydantic Integration Successful ---")
print(f"Name: {user.name}")
print(f"Role: {user.role}")
print(f"Experience: {user.years_experience} years")
print(f"Skills: {', '.join(user.skills)}")
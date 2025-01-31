import instructor
from litellm import completion
from litellm import completion, completion_cost, cost_per_token
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_litellm(completion)
instructor_resp, _ = client.chat.completions.create_with_completion(
    model="claude-3-5-sonnet-20240620",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Extract Jason is 25 years old.",
        }
    ],
    response_model=User,
)
print(type(instructor_resp))
instructor_cost = completion_cost(completion_response=instructor_resp, model="claude-3-opus-20240229")
print("Instructor cost: ", instructor_cost)

litellm_resp = completion(
    model="claude-3-5-sonnet-20240620",
    messages=[
        {"role": "user", "content": "Extract Jason is 25 years old."}
    ],
    max_tokens=1024,
)
litellm_cost = completion_cost(completion_response=litellm_resp, model="claude-3-opus-20240229")
print("Litellm cost: ", litellm_cost)
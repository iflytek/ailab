import openai
from langchain.chains.conversation.memory import ConversationalBufferWindowMemory

a = ConversationalBufferWindowMemory(k=10)



# doc is here https://platform.openai.com/docs/guides/chat/chat-vs-completions?utm_medium=email&_hsmi=248334739&utm_content=248334739&utm_source=hs_email
# chat_completion = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     api_key="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",
#     messages=[
#         # system message first, it helps set the behavior of the assistant
#         {"role": "system", "content": "You are a helpful assistant."},
#         # I am the user, and this is my prompt
#         {"role": "user", "content": "What's the best star wars movie?"},
#         # we can also add the previous conversation
#         # {"role": "assistant", "content": "Episode III."},
#     ],
# )
#
# print(chat_completion.choices[0].message.content)
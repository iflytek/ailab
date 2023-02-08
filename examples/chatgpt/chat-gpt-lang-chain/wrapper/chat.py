#!/usr/bin/env python
# coding:utf-8
""" 
@author: nivic ybyang7
@license: Apache Licence 
@file: chat
@time: 2023/02/08
@contact: ybyang7@iflytek.com
@site:  
@software: PyCharm 

# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛ 
"""

#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.
import datetime
from threading import Lock

# UNCOMMENT TO USE WHISPER
import warnings
import whisper

from langchain import ConversationChain, LLMChain

from langchain.agents import load_tools, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from threading import Lock
import os
import ssl
from contextlib import closing
from typing import Optional, Tuple
# Console to variable
from io import StringIO
import sys
import re

from openai.error import AuthenticationError, InvalidRequestError, RateLimitError

# Pertains to Express-inator functionality
from langchain.prompts import PromptTemplate

from polly_utils import PollyVoiceData, NEURAL_ENGINE
from azure_utils import AzureVoiceData

# Pertains to question answering functionality
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain

news_api_key = os.environ.get("NEWS_API_KEY",None)
tmdb_bearer_token = os.environ.get("TMDB_BEARER_TOKEN",None)

TOOLS_LIST = ['serpapi', 'wolfram-alpha', 'pal-math', 'pal-colored-objects', 'news-api', 'tmdb-api',
              'open-meteo-api']  # 'google-search'
# TOOLS_DEFAULT_LIST = ['serpapi', 'pal-math','wolfram-alpha']
TOOLS_DEFAULT_LIST = []
BUG_FOUND_MSG = "Congratulations, you've found a bug in this application!"
# AUTH_ERR_MSG = "Please paste your OpenAI key from openai.com to use this application. It is not necessary to hit a button or key after pasting it."
AUTH_ERR_MSG = "Please paste your OpenAI key from openai.com to use this application. "
MAX_TOKENS = 512

LOOPING_TALKING_HEAD = "videos/Masahiro.mp4"
TALKING_HEAD_WIDTH = "192"
MAX_TALKING_HEAD_TEXT_LENGTH = 155

# Pertains to Express-inator functionality
NUM_WORDS_DEFAULT = 0
MAX_WORDS = 400
FORMALITY_DEFAULT = "N/A"
TEMPERATURE_DEFAULT = 0.5
EMOTION_DEFAULT = "N/A"
LANG_LEVEL_DEFAULT = "N/A"
TRANSLATE_TO_DEFAULT = "N/A"
LITERARY_STYLE_DEFAULT = "N/A"
PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["original_words", "num_words", "formality", "emotions", "lang_level", "translate_to",
                     "literary_style"],
    template="Restate {num_words}{formality}{emotions}{lang_level}{translate_to}{literary_style}the following: \n{original_words}\n",
)

POLLY_VOICE_DATA = PollyVoiceData()
AZURE_VOICE_DATA = AzureVoiceData()

# Pertains to WHISPER functionality
WHISPER_DETECT_LANG = "Detect language"

# UNCOMMENT TO USE WHISPER
warnings.filterwarnings("ignore")
WHISPER_MODEL = whisper.load_model("tiny")
print("WHISPER_MODEL", WHISPER_MODEL)


########


def set_openai_api_key(api_key):
    """Set the api key and return chain.
    If no api_key, then None is returned.
    """
    if api_key and api_key.startswith("sk-") and len(api_key) > 50:
        os.environ["OPENAI_API_KEY"] = api_key
        print("\n\n ++++++++++++++ Setting OpenAI API key ++++++++++++++ \n\n")
        print(str(datetime.datetime.now()) + ": Before OpenAI, OPENAI_API_KEY length: " + str(
            len(os.environ["OPENAI_API_KEY"])))
        llm = OpenAI(temperature=0, max_tokens=MAX_TOKENS)
        print(str(datetime.datetime.now()) + ": After OpenAI, OPENAI_API_KEY length: " + str(
            len(os.environ["OPENAI_API_KEY"])))
        chain, express_chain, memory = load_chain(TOOLS_DEFAULT_LIST, llm)

        # Pertains to question answering functionality
        embeddings = OpenAIEmbeddings()
        qa_chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

        print(str(datetime.datetime.now()) + ": After load_chain, OPENAI_API_KEY length: " + str(
            len(os.environ["OPENAI_API_KEY"])))
        os.environ["OPENAI_API_KEY"] = ""
        return chain, express_chain, llm, embeddings, qa_chain, memory
    return None, None, None, None, None, None


def load_chain(tools_list, llm):
    chain = None
    express_chain = None
    memory = None
    if llm:
        print("\ntools_list", tools_list)
        tool_names = tools_list
        tools = load_tools(tool_names, llm=llm, news_api_key=news_api_key, tmdb_bearer_token=tmdb_bearer_token)

        memory = ConversationBufferMemory(memory_key="chat_history")

        chain = initialize_agent(tools, llm, agent="conversational-react-description", verbose=True, memory=memory)
        express_chain = LLMChain(llm=llm, prompt=PROMPT_TEMPLATE, verbose=True)
    return chain, express_chain, memory


# Pertains to Express-inator functionality
def transform_text(desc, express_chain, num_words, formality,
                   anticipation_level, joy_level, trust_level,
                   fear_level, surprise_level, sadness_level, disgust_level, anger_level,
                   lang_level, translate_to, literary_style):
    num_words_prompt = ""
    if num_words and int(num_words) != 0:
        num_words_prompt = "using up to " + str(num_words) + " words, "

    # Change some arguments to lower case
    formality = formality.lower()
    anticipation_level = anticipation_level.lower()
    joy_level = joy_level.lower()
    trust_level = trust_level.lower()
    fear_level = fear_level.lower()
    surprise_level = surprise_level.lower()
    sadness_level = sadness_level.lower()
    disgust_level = disgust_level.lower()
    anger_level = anger_level.lower()

    formality_str = ""
    if formality != "n/a":
        formality_str = "in a " + formality + " manner, "

    # put all emotions into a list
    emotions = []
    if anticipation_level != "n/a":
        emotions.append(anticipation_level)
    if joy_level != "n/a":
        emotions.append(joy_level)
    if trust_level != "n/a":
        emotions.append(trust_level)
    if fear_level != "n/a":
        emotions.append(fear_level)
    if surprise_level != "n/a":
        emotions.append(surprise_level)
    if sadness_level != "n/a":
        emotions.append(sadness_level)
    if disgust_level != "n/a":
        emotions.append(disgust_level)
    if anger_level != "n/a":
        emotions.append(anger_level)

    emotions_str = ""
    if len(emotions) > 0:
        if len(emotions) == 1:
            emotions_str = "with emotion of " + emotions[0] + ", "
        else:
            emotions_str = "with emotions of " + ", ".join(emotions[:-1]) + " and " + emotions[-1] + ", "

    lang_level_str = ""
    if lang_level != LANG_LEVEL_DEFAULT:
        lang_level_str = "at a " + lang_level + " level, " if translate_to == TRANSLATE_TO_DEFAULT else ""

    translate_to_str = ""
    if translate_to != TRANSLATE_TO_DEFAULT:
        translate_to_str = "translated to " + (
            "" if lang_level == TRANSLATE_TO_DEFAULT else lang_level + " level ") + translate_to + ", "

    literary_style_str = ""
    if literary_style != LITERARY_STYLE_DEFAULT:
        if literary_style == "Prose":
            literary_style_str = "as prose, "
        if literary_style == "Story":
            literary_style_str = "as a story, "
        elif literary_style == "Summary":
            literary_style_str = "as a summary, "
        elif literary_style == "Outline":
            literary_style_str = "as an outline numbers and lower case letters, "
        elif literary_style == "Bullets":
            literary_style_str = "as bullet points using bullets, "
        elif literary_style == "Poetry":
            literary_style_str = "as a poem, "
        elif literary_style == "Haiku":
            literary_style_str = "as a haiku, "
        elif literary_style == "Limerick":
            literary_style_str = "as a limerick, "
        elif literary_style == "Rap":
            literary_style_str = "as a rap, "
        elif literary_style == "Joke":
            literary_style_str = "as a very funny joke with a setup and punchline, "
        elif literary_style == "Knock-knock":
            literary_style_str = "as a very funny knock-knock joke, "
        elif literary_style == "FAQ":
            literary_style_str = "as a FAQ with several questions and answers, "

    formatted_prompt = PROMPT_TEMPLATE.format(
        original_words=desc,
        num_words=num_words_prompt,
        formality=formality_str,
        emotions=emotions_str,
        lang_level=lang_level_str,
        translate_to=translate_to_str,
        literary_style=literary_style_str
    )

    trans_instr = num_words_prompt + formality_str + emotions_str + lang_level_str + translate_to_str + literary_style_str
    if express_chain and len(trans_instr.strip()) > 0:
        generated_text = express_chain.run(
            {'original_words': desc, 'num_words': num_words_prompt, 'formality': formality_str,
             'emotions': emotions_str, 'lang_level': lang_level_str, 'translate_to': translate_to_str,
             'literary_style': literary_style_str}).strip()
    else:
        print("Not transforming text")
        generated_text = desc

    # replace all newlines with <br> in generated_text
    generated_text = generated_text.replace("\n", "\n\n")

    prompt_plus_generated = "GPT prompt: " + formatted_prompt + "\n\n" + generated_text

    print("\n==== date/time: " + str(datetime.datetime.now() - datetime.timedelta(hours=5)) + " ====")
    print("prompt_plus_generated: " + prompt_plus_generated)

    return generated_text


def run_chain(chain, inp, capture_hidden_text):
    output = ""
    hidden_text = None
    if capture_hidden_text:
        error_msg = None
        tmp = sys.stdout
        hidden_text_io = StringIO()
        sys.stdout = hidden_text_io

        try:
            output = chain.run(input=inp)
        except AuthenticationError as ae:
            error_msg = AUTH_ERR_MSG + str(datetime.datetime.now()) + ". " + str(ae)
            print("error_msg", error_msg)
        except RateLimitError as rle:
            error_msg = "\n\nRateLimitError: " + str(rle)
        except ValueError as ve:
            error_msg = "\n\nValueError: " + str(ve)
        except InvalidRequestError as ire:
            error_msg = "\n\nInvalidRequestError: " + str(ire)
        except Exception as e:
            error_msg = "\n\n" + BUG_FOUND_MSG + ":\n\n" + str(e)

        sys.stdout = tmp
        hidden_text = hidden_text_io.getvalue()

        # remove escape characters from hidden_text
        hidden_text = re.sub(r'\x1b[^m]*m', '', hidden_text)

        # remove "Entering new AgentExecutor chain..." from hidden_text
        hidden_text = re.sub(r"Entering new AgentExecutor chain...\n", "", hidden_text)

        # remove "Finished chain." from hidden_text
        hidden_text = re.sub(r"Finished chain.", "", hidden_text)

        # Add newline after "Thought:" "Action:" "Observation:" "Input:" and "AI:"
        hidden_text = re.sub(r"Thought:", "\n\nThought:", hidden_text)
        hidden_text = re.sub(r"Action:", "\n\nAction:", hidden_text)
        hidden_text = re.sub(r"Observation:", "\n\nObservation:", hidden_text)
        hidden_text = re.sub(r"Input:", "\n\nInput:", hidden_text)
        hidden_text = re.sub(r"AI:", "\n\nAI:", hidden_text)

        if error_msg:
            hidden_text += error_msg

        print("hidden_text: ", hidden_text)
    else:
        try:
            output = chain.run(input=inp)
        except AuthenticationError as ae:
            output = AUTH_ERR_MSG + str(datetime.datetime.now()) + ". " + str(ae)
            print("output", output)
        except RateLimitError as rle:
            output = "\n\nRateLimitError: " + str(rle)
        except ValueError as ve:
            output = "\n\nValueError: " + str(ve)
        except InvalidRequestError as ire:
            output = "\n\nInvalidRequestError: " + str(ire)
        except Exception as e:
            output = "\n\n" + BUG_FOUND_MSG + ":\n\n" + str(e)

    return output, hidden_text


class ChatWrapper:

    def __init__(self):
        self.lock = Lock()

    def reset_memory(self, history, memory):
        memory.clear()
        history = []
        return history, history, memory

    def __call__(
            self, api_key: str, inp: str, history: Optional[Tuple[str, str]], chain: Optional[ConversationChain],
            trace_chain: bool, speak_text: bool, talking_head: bool, monologue: bool, express_chain: Optional[LLMChain],
            num_words, formality, anticipation_level, joy_level, trust_level,
            fear_level, surprise_level, sadness_level, disgust_level, anger_level,
            lang_level, translate_to, literary_style, qa_chain, docsearch, use_embeddings
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            print("\n==== date/time: " + str(datetime.datetime.now()) + " ====")
            print("inp: " + inp)
            print("trace_chain: ", trace_chain)
            print("speak_text: ", speak_text)
            print("talking_head: ", talking_head)
            print("monologue: ", monologue)
            history = history or []
            # If chain is None, that is because no API key was provided.
            output = "Please paste your OpenAI key from openai.com to use this app. " + str(datetime.datetime.now())
            hidden_text = output

            if chain:
                # Set OpenAI key
                import openai
                openai.api_key = api_key
                if not monologue:
                    if use_embeddings:
                        if inp and inp.strip() != "":
                            if docsearch:
                                docs = docsearch.similarity_search(inp)
                                output = str(qa_chain.run(input_documents=docs, question=inp))
                            else:
                                output, hidden_text = "Please supply some text in the the Embeddings tab.", None
                        else:
                            output, hidden_text = "What's on your mind?", None
                    else:
                        output, hidden_text = run_chain(chain, inp, capture_hidden_text=trace_chain)
                else:
                    output, hidden_text = inp, None

            output = transform_text(output, express_chain, num_words, formality, anticipation_level, joy_level,
                                    trust_level,
                                    fear_level, surprise_level, sadness_level, disgust_level, anger_level,
                                    lang_level, translate_to, literary_style)

            text_to_display = output
            if trace_chain:
                text_to_display = hidden_text + "\n\n" + output
            history.append((inp, text_to_display))

            # html_video, temp_file, html_audio, temp_aud_file = None, None, None, None
            # if speak_text:
            #     if talking_head:
            #         if len(output) <= MAX_TALKING_HEAD_TEXT_LENGTH:
            #             html_video, temp_file = do_html_video_speak(output, translate_to)
            #         else:
            #             temp_file = LOOPING_TALKING_HEAD
            #             html_video = create_html_video(temp_file, TALKING_HEAD_WIDTH)
            #             html_audio, temp_aud_file = do_html_audio_speak(output, translate_to)
            #     else:
            #         html_audio, temp_aud_file = do_html_audio_speak(output, translate_to)
            # else:
            #     if talking_head:
            #         temp_file = LOOPING_TALKING_HEAD
            #         html_video = create_html_video(temp_file, TALKING_HEAD_WIDTH)
            #     else:
            #         # html_audio, temp_aud_file = do_html_audio_speak(output, translate_to)
            #         # html_video = create_html_video(temp_file, "128")
            #         pass

        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history, None, None, None, None, ""
        # return history, history, html_audio, temp_aud_file, ""


if __name__ == '__main__':
    api_key = "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"
    chain_state, express_chain_state, llm_state, embeddings_state, qa_chain_state, memory_state = set_openai_api_key(
        api_key)
    chat = ChatWrapper()
    history_state = []
    trace_chain_state = False
    speak_text_state = False
    talking_head_state = True
    monologue_state = False
    express_chain_state = None
    num_words_state = 0
    formality_state = "N/A"
    anticipation_level_state = "N/A"
    joy_level_state = "N/A"
    trust_level_state = "N/A"
    fear_level_state = "N/A"
    surprise_level_state = "N/A"
    sadness_level_state = "N/A"
    disgust_level_state = "N/A"
    anger_level_state = "N/A"
    lang_level_state = "N/A"
    translate_to_state = "N/A"
    literary_style_state = "N/A"
    docsearch_state = None
    use_embeddings_state = False

    chatbot, history_state, video_html, my_file, audio_html, tmp_aud_file, message = chat(api_key, "你是谁", history_state,
                                                                                          chain_state,
                                                                                          trace_chain_state,
                                                                                          speak_text_state,
                                                                                          talking_head_state,
                                                                                          monologue_state,
                                                                                          express_chain_state,
                                                                                          num_words_state,
                                                                                          formality_state,
                                                                                          anticipation_level_state,
                                                                                          joy_level_state,
                                                                                          trust_level_state,
                                                                                          fear_level_state,
                                                                                          surprise_level_state,
                                                                                          sadness_level_state,
                                                                                          disgust_level_state,
                                                                                          anger_level_state,
                                                                                          lang_level_state,
                                                                                          translate_to_state,
                                                                                          literary_style_state,
                                                                                          qa_chain_state,
                                                                                          docsearch_state,
                                                                                          use_embeddings_state)
    print(message)